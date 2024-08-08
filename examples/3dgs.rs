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
) -> anyhow::Result<(Tensor, Vec<usize>, Vec<usize>, Vec<usize>)> {
    let num_point = point2gauss.dims2()?.0;
    let point2gauss = point2gauss.storage_and_layout().0;
    let point2gauss = match point2gauss.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let mut point2splat = vec![0f32; num_point * NDOF_SPLAT];
    // transform points
    for i_point in 0..num_point {
        let pos_world = arrayref::array_ref![point2gauss, i_point * NDOF_GAUSS, 3];
        let rotdia = {
            let dia = arrayref::array_ref![point2gauss, i_point * NDOF_GAUSS + 7, 3];
            let quat = arrayref::array_ref![point2gauss, i_point * NDOF_GAUSS + 10, 4];
            let dia = del_geo_core::mat3_col_major::from_diagonal(dia);
            let rot = del_geo_core::quat::to_mat3_col_major(quat);
            let rotdia = del_geo_core::mat3_col_major::mult_mat_col_major(&rot, &dia);
            nalgebra::Matrix3::<f32>::from_column_slice(&rotdia)
        };
        let pos_ndc = del_geo_core::mat4_col_major::transform_homogeneous(&mvp, pos_world).unwrap();
        // dbg!(i_point, pos_ndc);
        let pos_pix = [
            (pos_ndc[0] + 1.0) * 0.5 * (img_shape.0 as f32),
            (1.0 - pos_ndc[1]) * 0.5 * (img_shape.1 as f32),
        ];
        let mvp_jacob = {
            let pos_world = nalgebra::Vector3::<f32>::from_column_slice(pos_world);
            let mvp = nalgebra::Matrix4::<f32>::from_column_slice(mvp);
            del_geo_nalgebra::mat4::jacobian_transform(&mvp, &pos_world)
        };
        let ndc2pix = nalgebra::Matrix2x3::<f32>::new(
            0.5 * (img_shape.0 as f32),
            0.,
            0.,
            0.,
            -0.5 * (img_shape.1 as f32),
            0.,
        );
        // let w0 = ndc2pix * prj_jacob * modelview * rotdia;
        let w0 = ndc2pix * mvp_jacob * rotdia;
        let w0 = w0 * w0.transpose();
        let w0 = w0.try_inverse().unwrap();
        let w0 = [w0.m11, w0.m12, w0.m22];
        let aabb = del_geo_core::mat2_sym::aabb2(&w0);
        let aabb = del_geo_core::aabb2::scale(&aabb, 3.0);
        let aabb = del_geo_core::aabb2::translate(&aabb, &pos_pix);
        point2splat[i_point * NDOF_SPLAT..i_point * NDOF_SPLAT + 2].copy_from_slice(&pos_pix);
        point2splat[i_point * NDOF_SPLAT + 2..i_point * NDOF_SPLAT + 5].copy_from_slice(&w0);
        point2splat[i_point * NDOF_SPLAT + 5..i_point * NDOF_SPLAT + 9].copy_from_slice(&aabb);
        point2splat[i_point * NDOF_SPLAT + 9] = -pos_ndc[2];
    }
    let mut idx2tilegauss: Vec<(u32, f32)> = vec![];
    let mut idx2point: Vec<usize> = vec![];
    const TILE_SIZE: usize = 16;
    let tile_shape: (usize, usize) = (img_shape.0 / TILE_SIZE, img_shape.1 / TILE_SIZE);
    for i_point in 0..num_point {
        let aabb = arrayref::array_ref![point2splat, i_point * NDOF_SPLAT + 5, 4];
        let depth = point2splat[i_point * NDOF_SPLAT + 9];
        let ix0 = (aabb[0] / TILE_SIZE as f32).floor() as i32;
        let iy0 = (aabb[1] / TILE_SIZE as f32).floor() as i32;
        let ix1 = (aabb[2] / TILE_SIZE as f32).floor() as i32 + 1;
        let iy1 = (aabb[3] / TILE_SIZE as f32).floor() as i32 + 1;
        let mut tiles = std::collections::BTreeSet::<usize>::new();
        for ix in ix0..ix1 {
            assert_ne!(ix, ix1);
            if ix < 0 || ix >= (tile_shape.0 as i32) {
                continue;
            }
            let ix = ix as usize;
            for iy in iy0..iy1 {
                assert_ne!(iy, iy1);
                if iy < 0 || iy >= (tile_shape.1 as i32) {
                    continue;
                }
                let iy = iy as usize;
                let i_tile = iy * tile_shape.0 + ix;
                tiles.insert(i_tile);
            }
        }
        for i_tile in tiles {
            idx2tilegauss.push((i_tile as u32, depth));
            idx2point.push(i_point);
        }
    }
    let num_idx = idx2tilegauss.len();
    let mut jdx2idx: Vec<usize> = (0..num_idx).collect();
    jdx2idx.sort_by(|&idx0, &idx1| {
        let itile0 = idx2tilegauss[idx0].0;
        let itile1 = idx2tilegauss[idx1].0;
        if itile0 != itile1 {
            itile0.cmp(&itile1)
        } else {
            let depth0 = idx2tilegauss[idx0].1;
            let depth1 = idx2tilegauss[idx1].1;
            depth0.partial_cmp(&depth1).unwrap()
        }
    });
    let num_tile = tile_shape.0 * tile_shape.1;
    let mut tile2jdx = vec![0usize; num_tile + 1];
    {
        for jdx in 0..jdx2idx.len() {
            let idx0 = jdx2idx[jdx];
            let i_tile = idx2tilegauss[idx0].0 as usize;
            tile2jdx[i_tile + 1] += 1;
        }
        for i_tile in 0..num_tile {
            tile2jdx[i_tile + 1] += tile2jdx[i_tile];
        }
    }
    /*
    for i_tile in 0..num_tile {
        for &idx in &jdx2idx[tile2jdx[i_tile]..tile2jdx[i_tile+1]] {
            let i_point0 = idx2point[idx];
            println!("{} {} {} {}", i_tile, idx2tilegauss[idx].0, idx2tilegauss[idx].1, point2splat[i_point0*NDOF_SPLAT+9]);
        }
    }
     */
    /*
    for jdx in 0..jdx2idx.len() {
        let idx0 = jdx2idx[jdx];
        let i_point0 = idx2point[idx0];
        println!("{} {} {} {}", jdx, idx2tilegauss[idx0].0, idx2tilegauss[idx0].1, point2splat[i_point0*NDOF_SPLAT+9]);
    }
     */
    let point2splat = Tensor::from_slice(&point2splat, (num_point, NDOF_SPLAT), &Device::Cpu)?;
    return Ok((point2splat, tile2jdx, jdx2idx, idx2point));
}

fn main() -> anyhow::Result<()> {
    let point2gauss = {
        let (tri2vtx, vtx2xyz, _vtx2uv) = {
            let mut obj = del_msh_core::io_obj::WavefrontObj::<usize, f32>::new();
            obj.load("examples/asset/spot_triangulated.obj")?;
            obj.unified_xyz_uv_as_trimesh()
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
            0.5,
            3.0,
        );
        let modelview =
            del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.], 0., 0., 0.);
        Camera {
            img_shape,
            projection,
            modelview,
        }
    };

    let img_trg = del_candle::load_img_as_tensor("examples/asset/trg0.png");
    // dbg!(img_trg.shape().dims3()?);
    // dbg!(img_trg.flatten_all()?.to_vec1::<f32>());

    for i_itr in 0..100 {
        let img_out = {
            let now = std::time::Instant::now();
            let mvp = del_geo_core::mat4_col_major::multmat(&cam.projection, &cam.modelview);
            let (point2splat, tile2jdx, jdx2idx, idx2point) =
                point_to_splat(point2gauss.as_detached_tensor(), &mvp, cam.img_shape)?;
            let render = del_candle::gaussian_splatting::Layer {
                point2splat: point2splat.clone(),
                img_shape: cam.img_shape,
                tile2jdx: tile2jdx.clone(),
                jdx2idx: jdx2idx.clone(),
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
            del_canvas::write_png_from_float_image_rgb(
                format!("target/points3d_gaussian_{}.png", i_itr),
                &cam.img_shape,
                &img_data,
            );
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
        let num_point = point2gauss.dims2()?.0;
        let dw_point2gauss = dw_point2gauss.flatten_all()?.to_vec1::<f32>()?;
        let mut point2gauss1 = point2gauss.flatten_all()?.to_vec1::<f32>()?;
        let lr = 0.000005;
        for i_point in 0..num_point {
            let delta = &dw_point2gauss[i_point * NDOF_GAUSS..(i_point + 1) * NDOF_GAUSS];
            let gauss0 =
                Vec::<f32>::from(&point2gauss1[i_point * NDOF_GAUSS..(i_point + 1) * NDOF_GAUSS]);
            let gauss1 = &mut point2gauss1[i_point * NDOF_GAUSS..(i_point + 1) * NDOF_GAUSS];
            for i in 0..7 {
                gauss1[i] -= delta[i] * lr;
            }
            {
                // 7-10
                let ddia = arrayref::array_ref![delta, 7, 3];
                let dia1 = arrayref::array_mut_ref![gauss1, 7, 3];
                for i in 0..3 {
                    let d = dia1[i] - ddia[i] * lr;
                    dia1[i] = d.max(1.0e-5f32);
                }
            }
            {
                // 10-14
                let quat0 = arrayref::array_ref![gauss0, 10, 4];
                let r0 = del_geo_core::quat::to_mat3_col_major(quat0);
                let daa = arrayref::array_ref![delta, 10, 3];
                let daa = del_geo_core::vec3::scaled(daa, lr);
                let dr = del_geo_core::vec3::to_mat3_from_axisangle_vec(&daa);
                let r1 = del_geo_core::mat3_col_major::mult_mat_col_major(&dr, &r0);
                let quat1 = del_geo_core::mat3_col_major::to_quaternion(&r1);
                let quat1 = del_geo_core::quat::normalized(&quat1);
                gauss1[10..14].copy_from_slice(&quat1);
            }
        }
        let point2gauss1 = Tensor::from_vec(point2gauss1, (num_point, NDOF_GAUSS), &Device::Cpu)?;
        point2gauss.set(&point2gauss1)?;
        println!("   update: {:.2?}", now.elapsed());
    }

    Ok(())
}
