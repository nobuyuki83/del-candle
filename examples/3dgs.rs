use candle_core::{Device, Tensor};
use rand::Rng;
use std::ops::Deref;

const NDOF_GAUSS: usize = 14; // xyz, rgba, s0,s1,s2, q0,q1,q2,q3
const NDOF_SPLAT: usize = 10; // pos_pix(2) + abc(3) + aabb(4) + ndc_z(1)

struct Camera {
    img_shape: (usize, usize),
    projection: [f32; 16],
    modelview: [f32; 16],
}

fn point_to_splat(
    point2gauss: Tensor,
    mvp: &[f32; 16],
    img_shape: (usize, usize),
) -> anyhow::Result<(Tensor, Vec<usize>, Vec<usize>)> {
    let num_point = point2gauss.dims2()?.0;
    let point2gauss = point2gauss.storage_and_layout().0;
    let point2gauss = match point2gauss.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let mut point2splat = vec![0f32; num_point * NDOF_SPLAT];
    // transform points
    for i_point in 0..num_point {
        let gauss = del_canvas_cpu::splat_gaussian2::Gauss::new(arrayref::array_ref![
            point2gauss,
            i_point * NDOF_GAUSS,
            NDOF_GAUSS
        ]);
        let pos_world = gauss.pos_world();
        let pos_ndc = del_geo_core::mat4_col_major::transform_homogeneous(&mvp, pos_world).unwrap();
        let pos_pix = [
            (pos_ndc[0] + 1.0) * 0.5 * (img_shape.0 as f32),
            (1.0 - pos_ndc[1]) * 0.5 * (img_shape.1 as f32),
        ];
        let w0 = gauss.sigma2inv(mvp, &img_shape);
        let aabb = del_geo_core::mat2_sym::aabb2(&w0);
        let aabb = del_geo_core::aabb2::scale(&aabb, 3.0);
        let aabb = del_geo_core::aabb2::translate(&aabb, &pos_pix);
        point2splat[i_point * NDOF_SPLAT..i_point * NDOF_SPLAT + 2].copy_from_slice(&pos_pix);
        point2splat[i_point * NDOF_SPLAT + 2..i_point * NDOF_SPLAT + 5].copy_from_slice(&w0);
        point2splat[i_point * NDOF_SPLAT + 5..i_point * NDOF_SPLAT + 9].copy_from_slice(&aabb);
        point2splat[i_point * NDOF_SPLAT + 9] = -pos_ndc[2];
    }
    let point2aabbdepth = |i_point: usize| {
        let aabb = arrayref::array_ref![point2splat, i_point * NDOF_SPLAT + 5, 4];
        let depth = point2splat[i_point * NDOF_SPLAT + 9];
        (aabb.clone(), depth)
    };
    let (tile2idx, idx2point) =
        del_canvas_cpu::tile_acceleration::tile2pnt(num_point, point2aabbdepth, img_shape, 16);
    let point2splat = Tensor::from_slice(&point2splat, (num_point, NDOF_SPLAT), &Device::Cpu)?;
    return Ok((point2splat, tile2idx, idx2point));
}

fn main() -> anyhow::Result<()> {
    let point2gauss = {
        let (tri2vtx, vtx2xyz) = {
            let mut obj = del_msh_core::io_obj::WavefrontObj::<usize, f32>::new();
            obj.load("examples/asset/spot_triangulated.obj")?;
            let (tri2vtx, vtx2xyz, _vtx2uv) = obj.unified_xyz_uv_as_trimesh();
            // let rot90x = del_geo_core::mat4_col_major::rot_x(std::f32::consts::PI * 0.5);
            //let vtx2xyz = del_msh_core::vtx2xyz::transform(&vtx2xyz, &rot90x);
            (tri2vtx, vtx2xyz)
        };
        const NUM_POINTS: usize = 10000;
        let mut pos2three: Vec<f32> = vec![0f32; NUM_POINTS * NDOF_GAUSS];
        let cumsumarea = del_msh_core::sampling::cumulative_area_sum(&tri2vtx, &vtx2xyz, 3);
        // let mut reng = rand::thread_rng();
        use rand::SeedableRng;
        let mut reng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        for i_point in 0..NUM_POINTS {
            let val01_a = reng.gen::<f32>();
            let val01_b = reng.gen::<f32>();
            let barycrd =
                del_msh_core::sampling::sample_uniformly_trimesh(&cumsumarea, val01_a, val01_b);
            let tri = del_msh_core::trimesh3::to_tri3(barycrd.0, &tri2vtx, &vtx2xyz);
            let pos_world = tri.position_from_barycentric_coordinates(barycrd.1, barycrd.2);
            let frame_z = del_geo_core::vec3::normalized(&tri.normal());
            let rgba = [
                frame_z[0] * 0.5 + 0.5,
                frame_z[1] * 0.5 + 0.5,
                frame_z[2] * 0.5 + 0.5,
                1.0,
            ];
            let d = [0.03, 0.03, 0.003];
            let (frame_x, frame_y) = del_geo_core::vec3::basis_xy_from_basis_z(&frame_z);
            let r = del_geo_core::mat3_col_major::from_column_vectors(&frame_x, &frame_y, &frame_z);
            let q = del_geo_core::mat3_col_major::to_quaternion(&r);
            pos2three[i_point * NDOF_GAUSS..i_point * NDOF_GAUSS + 3].copy_from_slice(&pos_world);
            pos2three[i_point * NDOF_GAUSS + 3..i_point * NDOF_GAUSS + 7].copy_from_slice(&rgba);
            pos2three[i_point * NDOF_GAUSS + 7..i_point * NDOF_GAUSS + 10].copy_from_slice(&d);
            pos2three[i_point * NDOF_GAUSS + 10..i_point * NDOF_GAUSS + 14].copy_from_slice(&q);
        }
        Tensor::from_vec(pos2three, (NUM_POINTS, NDOF_GAUSS), &Device::Cpu)?
    };
    let point2gauss = candle_core::Var::from_tensor(&point2gauss)?;

    const TILE_SIZE: usize = 16;
    let cam = {
        let img_shape = (TILE_SIZE * 28, TILE_SIZE * 28);
        let projection = del_geo_core::mat4_col_major::camera_perspective_blender(
            img_shape.0 as f32 / img_shape.1 as f32,
            24f32,
            //50f32,
            0.3,
            3.0,
            true
        );
        let modelview =
            //del_geo_core::mat4_col_major::camera_external_blender(&[-1.8, 2., 1.3], 65., 0., 222.);
        del_geo_core::mat4_col_major::camera_external_blender(&[-0., 0., 2.], 0., 0., 0.);
        Camera {
            img_shape,
            projection,
            modelview,
        }
    };

    let img_trg = del_candle::load_img_as_tensor("examples/asset/trg0.png")?;
    //let img_trg = del_candle::load_img_as_tensor("examples/asset/trg0.png");
    // dbg!(img_trg.shape().dims3()?);
    // dbg!(img_trg.flatten_all()?.to_vec1::<f32>());

    for i_itr in 0..100 {
        //if i_itr == 19 {
        // println!("after update gauss min {:?}",point2gauss.min(0)?.to_vec1::<f32>()?);
        // println!("after update gauss max {:?}",point2gauss.max(0)?.to_vec1::<f32>()?);
        let img_out = {
            let now = std::time::Instant::now();
            let mvp = del_geo_core::mat4_col_major::mult_mat(&cam.projection, &cam.modelview);
            let (point2splat, tile2idx, idx2point) =
                point_to_splat(point2gauss.as_detached_tensor(), &mvp, cam.img_shape)?;
            // println!("after update splat min {:?}",point2splat.min(0)?.to_vec1::<f32>()?);
            // println!("after update splat max {:?}",point2splat.max(0)?.to_vec1::<f32>()?);
            println!("   precomp: {:.2?}", now.elapsed());
            let now = std::time::Instant::now();
            let render = del_candle::gaussian_splatting::Layer {
                point2splat: point2splat.clone(),
                img_shape: cam.img_shape,
                tile2idx: tile2idx.clone(),
                idx2point: idx2point.clone(),
                tile_size: TILE_SIZE,
                mvp: mvp.clone(),
            };
            let img = point2gauss.apply_op1(render)?;
            println!("   render: {:.2?}", now.elapsed());
            img
        };
        {
            let img_data = img_out.flatten_all()?.to_vec1::<f32>()?;
            del_canvas_image::write_png_from_float_image_rgb(
                format!("target/points3d_gaussian_{}.png", i_itr),
                &cam.img_shape,
                &img_data,
            )?;
        }
        let diff = img_trg.sub(&img_out).unwrap().sqr()?.sum_all()?;
        println!("{} {}", i_itr, diff.to_vec0::<f32>()?);
        let grad = {
            let now = std::time::Instant::now();
            let grad = diff.backward()?;
            println!("   backward: {:.2?}", now.elapsed());
            grad
        };
        let now = std::time::Instant::now();
        let dw_point2gauss = grad.get(&point2gauss).unwrap();
        // dbg!( dw_point2gauss.min(0)?.to_vec1::<f32>()?);
        // dbg!( dw_point2gauss.max(0)?.to_vec1::<f32>()?);
        let num_point = point2gauss.dims2()?.0;
        let dw_point2gauss = dw_point2gauss.flatten_all()?.to_vec1::<f32>()?;
        let mut point2gauss1 = point2gauss.flatten_all()?.to_vec1::<f32>()?;
        let lr = 0.000005;
        //let lr = 0.001;
        for i_point in 0..num_point {
            let delta = &dw_point2gauss[i_point * NDOF_GAUSS..(i_point + 1) * NDOF_GAUSS];
            let gauss0 =
                Vec::<f32>::from(&point2gauss1[i_point * NDOF_GAUSS..(i_point + 1) * NDOF_GAUSS]);
            let gauss1 = &mut point2gauss1[i_point * NDOF_GAUSS..(i_point + 1) * NDOF_GAUSS];
            for i in 0..7 {
                gauss1[i] -= delta[i] * lr;
            }
            {
                // 7-10 (dia)
                let ddia = arrayref::array_ref![delta, 7, 3];
                let dia1 = arrayref::array_mut_ref![gauss1, 7, 3];
                for i in 0..3 {
                    let d = dia1[i] - ddia[i] * lr;
                    dia1[i] = d.max(1.0e-5f32);
                }
            }
            {
                // 10-14 (rot)
                let quat0 = arrayref::array_ref![gauss0, 10, 4];
                let daa = arrayref::array_ref![delta, 10, 3];
                let daa = del_geo_core::vec3::scaled(daa, lr);
                let dq = del_geo_core::vec3::to_quaternion_from_axis_angle_vector(&daa);
                let quat1 = del_geo_core::quaternion::mult_quaternion(&dq, &quat0);
                //let r0 = del_geo_core::quat::to_mat3_col_major(quat0);
                //let dr = del_geo_core::vec3::to_mat3_from_axisangle_vec(&daa);
                //let r1 = del_geo_core::mat3_col_major::mult_mat_col_major(&dr, &r0);
                //let quat1 = del_geo_core::mat3_col_major::to_quaternion(&r1);
                let quat1 = del_geo_core::quaternion::normalized(&quat1);
                gauss1[10..14].copy_from_slice(&quat1);
            }
        }
        let point2gauss1 = Tensor::from_vec(point2gauss1, (num_point, NDOF_GAUSS), &Device::Cpu)?;
        point2gauss.set(&point2gauss1)?;
        println!("   update: {:.2?}", now.elapsed());
    }

    Ok(())
}
