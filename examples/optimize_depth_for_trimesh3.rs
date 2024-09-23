use candle_core::Tensor;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 32, 32);
    println!(
        "num_tri: {},  num_vtx: {}",
        tri2vtx.len() / 3,
        vtx2xyz.len() / 3
    );
    let lambda = 1.;
    let mut ls = {
        let tri2vtx: Vec<usize> = tri2vtx.iter().map(|v| *v as usize).collect();
        let ls =
            del_fem_core::laplace_tri3::to_linearsystem(&tri2vtx, vtx2xyz.len() / 3, 1., lambda);
        ls
    };

    let num_tri = tri2vtx.len() / 3;
    let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &candle_core::Device::Cpu)?;
    let num_vtx = vtx2xyz.len() / 3;
    let vtx2xyz = candle_core::Var::from_vec(vtx2xyz, (num_vtx, 3), &candle_core::Device::Cpu)?;
    let img_shape = (300, 300);
    //
    let transform_ndc2world = del_geo_core::mat4_col_major::identity::<f32>();
    let (pix2depth_trg, pix2mask) = {
        let mut img2depth_trg = vec![0f32; img_shape.0 * img_shape.1];
        let mut img2mask = vec![0f32; img_shape.0 * img_shape.1];
        for i_h in 0..img_shape.1 {
            for i_w in 0..img_shape.0 {
                let (ray_org, _ray_dir) = del_canvas_cpu::cam3::ray3_homogeneous(
                    (i_w, i_h),
                    &img_shape,
                    &transform_ndc2world,
                );
                let x = ray_org[0];
                let y = ray_org[1];
                let r = (x * x + y * y).sqrt();
                if r > 0.5 {
                    continue;
                }
                img2depth_trg[i_h * img_shape.0 + i_w] = 0.6;
                img2mask[i_h * img_shape.0 + i_w] = 1.0;
            }
        }
        let img2depth_trg = Tensor::from_vec(img2depth_trg, img_shape, &candle_core::Device::Cpu)?;
        let img2mask = Tensor::from_vec(img2mask, img_shape, &candle_core::Device::Cpu)?;
        (img2depth_trg, img2mask)
    };
    {
        let pix2depth_trg = pix2depth_trg.flatten_all()?.to_vec1::<f32>()?;
        del_canvas_cpu::write_png_from_float_image_grayscale(
            "target/pix2depth_trg.png",
            &img_shape,
            &pix2depth_trg,
        )?;
        //
        let pix2mask = pix2mask.flatten_all()?.to_vec1::<f32>()?;
        del_canvas_cpu::write_png_from_float_image_grayscale(
            "target/pix2mask.png",
            &img_shape,
            &pix2mask,
        )?;
    }

    let now = Instant::now();
    for _itr in 0..100 {
        let pix2depth = del_candle::render_meshtri3_depth::render(
            &tri2vtx,
            &vtx2xyz,
            &img_shape,
            &transform_ndc2world,
        )?;
        assert_eq!(pix2depth.shape().dims2()?, (img_shape.1, img_shape.0));
        /*
        {
            let pix2depth = pix2depth.flatten_all()?.to_vec1::<f32>()?;
            del_canvas::write_png_from_float_image("target/pix2depth.png", &img_shape, &pix2depth);
            //let pix2diff = pix2diff.flatten_all()?.to_vec1::<f32>()?;
            //del_canvas::write_png_from_float_image("target/pix2diff.png", &img_shape, &pix2diff);
        }
         */
        let pix2diff = pix2depth.sub(&pix2depth_trg)?.mul(&pix2mask)?;
        let loss = pix2diff.sqr()?.sum_all()?;
        println!("loss: {}", loss.to_vec0::<f32>()?);
        let grad = loss.backward()?;
        let dw_vtx2xyz = grad.get(&vtx2xyz).unwrap();
        let dw_vtx2xyz = {
            let x = dw_vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
            ls.r_vec = x;
            ls.solve_cg();
            Tensor::from_vec(ls.u_vec.clone(), (num_vtx, 3), &candle_core::Device::Cpu)?
        };
        let _ = vtx2xyz.set(&vtx2xyz.as_tensor().sub(&(dw_vtx2xyz * 0.001)?)?);
    }

    {
        let vtx2xyz = vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
        let tri2vtx = tri2vtx.flatten_all()?.to_vec1::<u32>()?;
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz("target/hoge.obj", &tri2vtx, &vtx2xyz, 3)?;
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    Ok(())
}
