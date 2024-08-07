use candle_core::{Device, Tensor};
use rand::Rng;
use std::ops::Deref;

struct Camera {
    img_shape: (usize, usize),
    projection: [f32;16],
    modelview: [f32;16]
}

fn point_to_splat(point2gauss: Tensor, cam: &Camera) -> anyhow::Result<(Tensor,Vec<usize>,Vec<usize>,Vec<usize>)> {
    let transform_world2ndc =
        del_geo_core::mat4_col_major::multmat(&cam.projection, &cam.modelview);
    let modelview = nalgebra::Matrix3::<f32>::new(
        cam.modelview[0],
        cam.modelview[4],
        cam.modelview[8],
        cam.modelview[1],
        cam.modelview[5],
        cam.modelview[9],
        cam.modelview[2],
        cam.modelview[6],
        cam.modelview[10],
    );
    let num_point = point2gauss.dims2()?.0;
    let point2gauss = point2gauss.storage_and_layout().0;
    let point2gauss = match point2gauss.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    const NDOF_GAUSS: usize = 14; // xyz, rgba, s0,s1,s2, q0,q1,q2,q3
    const NDOF_SPLAT: usize = 10; // pos_pix(2) + abc(3) + aabb(4) + ndc_z(1)
    let mut point2splat = vec!(0f32; num_point* NDOF_SPLAT);
    // transform points
    for i_point in 0..num_point {
        let pos_world = arrayref::array_ref![point2gauss, i_point*NDOF_GAUSS, 3];
        let rotdia = {
            let dia = arrayref::array_ref![point2gauss, i_point*NDOF_GAUSS+7, 3];
            let quat = arrayref::array_ref![point2gauss, i_point*NDOF_GAUSS+10, 4];
            let dia = del_geo_core::mat3_col_major::from_diagonal(dia);
            let rot = del_geo_core::quat::to_mat3_col_major(quat);
            let rotdia = del_geo_core::mat3_col_major::mult_mat_col_major(&rot, &dia);
            nalgebra::Matrix3::<f32>::from_column_slice(&rotdia)
        };
        let pos_ndc = del_geo_core::mat4_col_major::transform_homogeneous(
            &transform_world2ndc,
            pos_world,
        ).unwrap();
        // dbg!(i_point, pos_ndc);
        let pos_pix = [
            (pos_ndc[0] + 1.0) * 0.5 * (cam.img_shape.0 as f32),
            (1.0 - pos_ndc[1]) * 0.5 * (cam.img_shape.1 as f32)];
        let prj_jacob = {
            let mv_world =
                del_geo_core::mat4_col_major::transform_homogeneous(&cam.modelview, &pos_world).unwrap();
            let mv_world = nalgebra::Vector3::<f32>::from_column_slice(&mv_world);
            let cam_projection = nalgebra::Matrix4::<f32>::from_column_slice(&cam.projection);
            del_geo_nalgebra::mat4::jacobian_transform(&cam_projection, &mv_world)
        };
        let ndc2pix = nalgebra::Matrix2x3::<f32>::new(
            0.5 * (cam.img_shape.0 as f32),
            0.,
            0.,
            0.,
            -0.5 * (cam.img_shape.1 as f32),
            0.,
        );
        let w0 = ndc2pix * prj_jacob * modelview * rotdia;
        let w0 = w0 * w0.transpose();
        let w0 = w0.try_inverse().unwrap();
        let w0 = [w0.m11, w0.m12, w0.m22];
        let aabb = del_geo_core::mat2_sym::aabb2(&w0);
        let aabb = del_geo_core::aabb2::scale(&aabb, 3.0);
        let aabb = del_geo_core::aabb2::translate(&aabb, &pos_pix);
        point2splat[i_point* NDOF_SPLAT..i_point* NDOF_SPLAT +2].copy_from_slice(&pos_pix);
        point2splat[i_point* NDOF_SPLAT +2..i_point* NDOF_SPLAT +5].copy_from_slice(&w0);
        point2splat[i_point* NDOF_SPLAT +5..i_point* NDOF_SPLAT +9].copy_from_slice(&aabb);
        point2splat[i_point* NDOF_SPLAT +9] = -pos_ndc[2];
    }
    let mut idx2tilegauss : Vec<(u32,f32)> = vec!();
    let mut idx2point: Vec<usize> = vec!();
    const TILE_SIZE: usize = 16;
    let tile_shape: (usize, usize) = (cam.img_shape.0 / TILE_SIZE, cam.img_shape.1 / TILE_SIZE);
    for i_point in 0..num_point {
        let aabb = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT+5, 4];
        let depth = point2splat[i_point*NDOF_SPLAT+9];
        let ix0 = (aabb[0] / TILE_SIZE as f32).floor() as i32;
        let iy0 = (aabb[1] / TILE_SIZE as f32).floor() as i32;
        let ix1 = (aabb[2] / TILE_SIZE as f32).floor() as i32 + 1;
        let iy1 = (aabb[3] / TILE_SIZE as f32).floor() as i32 + 1;
        let mut tiles = std::collections::BTreeSet::<usize>::new();
        for ix in ix0..ix1 {
            assert_ne!(ix,ix1);
            if ix < 0 || ix >= (tile_shape.0 as i32) { continue; }
            let ix = ix as usize;
            for iy in iy0..iy1 {
                assert_ne!(iy,iy1);
                if iy < 0 || iy >= (tile_shape.1 as i32) { continue; }
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
        }
        else{
            let depth0 = idx2tilegauss[idx0].1;
            let depth1 = idx2tilegauss[idx1].1;
            depth0.partial_cmp(&depth1).unwrap()
        }
    });
    let num_tile = tile_shape.0 * tile_shape.1;
    let mut tile2jdx = vec!(0usize; num_tile + 1);
    {
        for jdx in 0..jdx2idx.len() {
            let idx0 = jdx2idx[jdx];
            let i_tile = idx2tilegauss[idx0].0 as usize;
            tile2jdx[i_tile+1] += 1;
        }
        for i_tile in 0..num_tile {
            tile2jdx[i_tile+1] += tile2jdx[i_tile];
        }
    }
    let point2splat = Tensor::from_slice(&point2splat, (num_point, NDOF_SPLAT), &Device::Cpu)?;
    return Ok((point2splat, tile2jdx, jdx2idx, idx2point));
}

fn main() -> anyhow::Result<()>{
    const TILE_SIZE: usize = 16;

    const NDOF_GAUSS: usize = 14; // xyz, rgba, s0,s1,s2, q0,q1,q2,q3
    const NDOF_SPLAT: usize = 10; // pos_pix(2) + abc(3) + aabb(4) + ndc_z(1)
    let point2gauss = {
        let (tri2vtx, vtx2xyz, vtx2uv) = {
            let mut obj = del_msh_core::io_obj::WavefrontObj::<usize, f32>::new();
            obj.load("examples/asset/spot_triangulated.obj")?;
            obj.unified_xyz_uv_as_trimesh()
        };
        const NUM_POINTS: usize = 10_000;
        let mut pos2three: Vec<f32> = vec![0f32; NUM_POINTS * NDOF_GAUSS];
        let cumsumarea = del_msh_core::sampling::cumulative_area_sum(&tri2vtx, &vtx2xyz, 3);
        // let mut reng = rand::thread_rng();
        use rand::SeedableRng;
        let mut reng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
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
                1.0];
            let d = [0.03, 0.03, 0.003];
            let (frame_x, frame_y) = del_geo_core::vec3::basis_xy_from_basis_z(&frame_z);
            let r = del_geo_core::mat3_col_major::from_column_vectors(&frame_x, &frame_y, &frame_z);
            let q = del_geo_core::mat3_col_major::to_quaternion(&r);
            pos2three[i_point * NDOF_GAUSS..i_point * NDOF_GAUSS + 3].copy_from_slice(&pos_world);
            pos2three[i_point * NDOF_GAUSS +3..i_point * NDOF_GAUSS + 7].copy_from_slice(&rgba);
            pos2three[i_point * NDOF_GAUSS +7..i_point * NDOF_GAUSS + 10].copy_from_slice(&d);
            pos2three[i_point * NDOF_GAUSS +10..i_point * NDOF_GAUSS + 14].copy_from_slice(&q);
        }
        candle_core::Tensor::from_vec(pos2three, (NUM_POINTS, NDOF_GAUSS), &Device::Cpu)?
    };

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
        Camera { img_shape, projection, modelview }
    };

    println!("point2splat");
    let (point2splat,tile2jdx, jdx2idx, idx2point)
        = point_to_splat(point2gauss.clone(), &cam)?;

    /*
    {
        println!("points");
        let num_point = point2gauss.dims2()?.0;
        let point2gauss = point2gauss.storage_and_layout().0;
        let point2gauss = match point2gauss.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let point2splat = point2splat.storage_and_layout().0;
        let point2splat = match point2splat.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };

        let img_size = cam.img_shape;
        let mut img_data = vec![0f32; img_size.1 * img_size.0 * 3];
        for i_point in idx2point.iter().rev() {
            let pos_pix = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT, 2];
            let color = arrayref::array_ref![point2gauss, i_point*NDOF_GAUSS+3, 3];
            let i_x = pos_pix[0] as usize;
            let i_y = pos_pix[1] as usize;
            img_data[(i_y * img_size.0 + i_x) * 3 + 0] = color[0];
            img_data[(i_y * img_size.0 + i_x) * 3 + 1] = color[1];
            img_data[(i_y * img_size.0 + i_x) * 3 + 2] = color[2];
        }
        del_canvas::write_png_from_float_image_rgb("target/points3d_pix.png", &img_size, &img_data);
    }
     */

    {
        println!("gaussian_naive");
        let point2gauss = point2gauss.storage_and_layout().0;
        let point2gauss = match point2gauss.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let point2splat = point2splat.storage_and_layout().0;
        let point2splat = match point2splat.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };

        let img_w = cam.img_shape.0;
        let img_h = cam.img_shape.1;
        let now = std::time::Instant::now();
        let mut img_data = vec![0f32; img_h * img_w * 3];
        for ih in 0..img_h {
            for iw in 0..img_w {
                let tile_shape: (usize, usize) = (cam.img_shape.0 / TILE_SIZE, cam.img_shape.1 / TILE_SIZE);
                let i_tile = (ih / TILE_SIZE) * tile_shape.0 + (iw / TILE_SIZE);
                let t = nalgebra::Vector2::<f32>::new(iw as f32 + 0.5, ih as f32 + 0.5);
                let mut alpha_sum = 0f32;
                let mut alpha_occu = 1f32;
                for &idx in &jdx2idx[tile2jdx[i_tile]..tile2jdx[i_tile+1]] {
                    let i_point = idx2point[idx];
                    let pos_pix = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT, 2];
                    let abc = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT+2, 3];
                    let aabb = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT+5, 4];
                    let w = nalgebra::Matrix2::<f32>::new(abc[0], abc[1], abc[1], abc[2]);
                    let pos_pix = nalgebra::Vector2::<f32>::from_column_slice(pos_pix);
                    let color = arrayref::array_ref![point2gauss, i_point*NDOF_GAUSS+3, 3];
                    if !del_geo_core::aabb2::is_inlcude_point(&aabb, &[t[0], t[1]]) { continue; }
                    let t0 = t - pos_pix;
                    let e = (t0.transpose() * w * t0).x;
                    let e = (-0.5 * e).exp();
                    let e_out = alpha_occu * e;
                    img_data[(ih * img_w + iw) * 3 + 0] += color[0] * e_out;
                    img_data[(ih * img_w + iw) * 3 + 1] += color[1] * e_out;
                    img_data[(ih * img_w + iw) * 3 + 2] += color[2] * e_out;
                    alpha_occu *= 1f32 - e;
                    alpha_sum += e_out;
                    if alpha_sum > 0.999 { break; }
                }
            }
        }
        del_canvas::write_png_from_float_image_rgb(
            "target/points3d_gaussian.png",
            &(img_w, img_h),
            &img_data,
        );
        println!("   Elapsed gaussian_naive: {:.2?}", now.elapsed());
    }

    /*
    {
        // visualize Gaussian as ellipsoid
        println!("gaussian_ellipse");
        let point2gauss = point2gauss.storage_and_layout().0;
        let point2gauss = match point2gauss.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let point2splat = point2splat.storage_and_layout().0;
        let point2splat = match point2splat.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let img_shape = cam.img_shape;
        let now = std::time::Instant::now();
        let mut img_data = vec![0f32; img_shape.1 * img_shape.0 * 3];
        for ih in 0..img_shape.1 {
            for iw in 0..img_shape.0 {
                let t = nalgebra::Vector2::<f32>::new(iw as f32 + 0.5, ih as f32 + 0.5);
                for i_point in idx2point.iter() {
                    let pos_pix = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT, 2];
                    let abc = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT+2, 3];
                    let aabb = arrayref::array_ref![point2splat, i_point*NDOF_SPLAT+5, 4];
                    let w = nalgebra::Matrix2::<f32>::new(abc[0], abc[1], abc[1], abc[2]);
                    let pos_pix = nalgebra::Vector2::<f32>::from_column_slice(pos_pix);
                    let color = arrayref::array_ref![point2gauss, i_point*NDOF_GAUSS+3, 3];
                    if !del_geo_core::aabb2::is_inlcude_point(&aabb, &[t[0], t[1]]) { continue }
                    let t0 = t - pos_pix;
                    let a = (t0.transpose() * w * t0).x;
                    if a > 1f32 {
                        continue;
                    }
                    img_data[(ih * img_shape.0 + iw) * 3 + 0] = color[0];
                    img_data[(ih * img_shape.0 + iw) * 3 + 1] = color[1];
                    img_data[(ih * img_shape.0 + iw) * 3 + 2] = color[2];
                    break;
                }
            }
        }
        del_canvas::write_png_from_float_image_rgb(
            "target/points3d_ellipse.png",
            &img_shape,
            &img_data,
        );
        println!("   Elapsed ellipse: {:.2?}", now.elapsed());
    }
     */


    Ok(())
}