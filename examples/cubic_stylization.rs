use del_msh::io_svg::{polybezier2polyloop, svg_loops_from_outline_path, svg_outline_path_from_shape};

fn rotate90(edge2xy: candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let x = edge2xy.get_on_dim(1, 0)?;
    let y = edge2xy.get_on_dim(1, 1)?;
    candle_core::Tensor::stack(&[(y * -1.)?,x], 1)
}

fn main() -> anyhow::Result<()>{
    let vtx2xy = {
        let str0 = "M2792 12789 c-617 -83 -1115 -568 -1244 -1212 -32 -160 -32 -443 0 \
    -602 76 -382 282 -720 573 -938 l36 -27 -58 -172 c-90 -269 -174 -590 -216 \
    -833 -13 -76 -17 -159 -17 -345 -1 -212 2 -261 21 -360 43 -225 113 -418 217 \
    -601 204 -356 574 -691 972 -880 96 -45 103 -51 160 -126 32 -43 105 -126 162 \
    -185 l103 -106 -47 -44 c-143 -131 -352 -391 -469 -584 -306 -501 -465 -1076 \
    -501 -1807 -5 -117 -9 -137 -23 -137 -38 0 -104 26 -211 85 -440 240 -827 302 \
    -1345 215 -216 -37 -301 -67 -409 -144 -258 -186 -410 -530 -476 -1081 -17 \
    -143 -24 -516 -11 -655 42 -486 188 -848 446 -1105 208 -209 459 -325 790 \
    -366 110 -13 513 -17 615 -6 l65 8 24 -63 c132 -354 523 -580 1149 -668 252 \
    -35 395 -44 722 -44 258 -1 351 3 450 17 134 19 295 54 400 89 74 23 256 107 \
    297 136 27 18 34 18 133 5 150 -19 624 -28 731 -14 84 11 86 11 150 -18 298 \
    -135 701 -204 1259 -218 280 -6 462 4 662 38 459 78 788 280 941 577 25 48 51 \
    106 58 130 9 31 16 41 28 37 41 -12 362 -26 491 -20 388 17 612 78 837 228 \
    336 223 534 574 615 1092 19 119 22 181 22 420 0 285 -12 437 -51 635 -14 73 \
    -20 89 -48 112 -24 20 -40 51 -65 120 -79 227 -184 405 -319 539 -130 130 \
    -226 178 -463 233 -188 44 -247 51 -438 50 -152 0 -203 -4 -286 -22 -199 -43 \
    -339 -101 -579 -239 -77 -44 -158 -86 -180 -92 -44 -12 -170 -14 -251 -5 l-51 \
    6 -6 257 c-8 352 -37 606 -102 896 -95 423 -268 810 -513 1146 l-41 56 39 38 \
    c37 36 40 37 127 42 478 24 909 263 1196 664 103 143 213 372 285 590 148 450 \
    215 839 216 1264 l1 230 65 32 c246 121 482 349 628 608 267 473 263 1087 -10 \
    1559 -215 371 -560 622 -978 712 -117 25 -398 26 -522 1 -200 -40 -417 -137 \
    -576 -257 -52 -38 -95 -70 -97 -70 -2 0 -45 30 -95 66 -389 280 -904 530 \
    -1298 629 -116 29 -289 57 -507 82 -229 26 -799 26 -1000 0 -265 -35 -499 -87 \
    -714 -159 l-124 -42 -35 44 c-75 95 -259 267 -350 328 -157 105 -323 175 -500 \
    212 -114 24 -350 34 -460 19z";
        let strs = svg_outline_path_from_shape(str0);
        let loops = svg_loops_from_outline_path(&strs);
        assert_eq!(loops.len(), 1);
        let vtx2xy = polybezier2polyloop(
            &loops[0].0, &loops[0].1, loops[0].2, 10.0);
        let vtx2xy = del_msh::vtx2xyz::from_array_of_nalgebra(&vtx2xy);
        del_msh::polyloop::resample::<_, 2>(&vtx2xy, 100)
    };
    let vtx2diff_ini = {
        let vtx2xy = candle_core::Tensor::from_slice(
            vtx2xy.as_slice(),
            candle_core::Shape::from((vtx2xy.len()/2, 2)),
            &candle_core::Device::Cpu).unwrap();
        let polyloop_to_diffcoord = del_candle::polyloop2_to_diffcoord::Layer {};
        vtx2xy.apply_op1(polyloop_to_diffcoord)?
    };
    let vtx2xy = candle_core::Var::from_slice(
        vtx2xy.as_slice(),
        candle_core::Shape::from((vtx2xy.len()/2, 2)),
        &candle_core::Device::Cpu).unwrap();
    for iter in 0..300 {
        let polyloop_to_edgevector = del_candle::polyloop_to_edgevector::Layer {};
        let edge2xy = vtx2xy.apply_op1(polyloop_to_edgevector)?;
        let edge2nrm = rotate90(edge2xy)?;
        /*
        let edge2len = edge2nrm.sqr().unwrap().sum(1)?.sqrt()?;
        let edge2len = candle_core::Tensor::stack(&[edge2len.clone(), edge2len.clone()], 1)?;
         let edge2unrm = edge2nrm.div(&edge2len)?;
        // dbg!(edge2unrm.sqr().unwrap().sum(1)?.sqrt()?.to_vec1::<f32>());
         */
        let mut edge2norm_trg = edge2nrm.flatten_all()?.to_vec1::<f32>()?;
        for norm in edge2norm_trg.chunks_mut(2) {
            let x0 = norm[0];
            let y0 = norm[1];
            let len = (x0*x0 + y0*y0).sqrt();
            if y0 > x0 && y0 > -x0 { norm[0] = 0f32; norm[1] = len; }
            if y0 < x0 && y0 < -x0 { norm[0] = 0f32; norm[1] = -len; }
            if y0 > x0 && y0 < -x0 { norm[0] = -len; norm[1] = 0f32; }
            if y0 < x0 && y0 > -x0 { norm[0] = len; norm[1] = 0f32; }
        }
        let edge2norm_trg = candle_core::Tensor::from_slice(
            edge2norm_trg.as_slice(),
            candle_core::Shape::from(vtx2xy.shape()),
            &candle_core::Device::Cpu).unwrap();
        let unorm_diff = edge2nrm.sub(&edge2norm_trg)?.sqr()?.sum_all()?;
        let polyloop_to_diffcoord = del_candle::polyloop2_to_diffcoord::Layer {};
        let magdiffc = vtx2xy.apply_op1(polyloop_to_diffcoord)?.sub(&vtx2diff_ini)?.sqr()?.sum_all()?;
        // let magdiffc = vtx2xy.apply_op1(polyloop_to_diffcoord)?.sqr().unwrap().sum_all()?;
        dbg!(unorm_diff.to_vec0::<f32>()?);
        //let loss = if iter < 100 { (magdiffc*2.0)? }
        //else{ (unorm_diff+magdiffc*0.3)? };
        let loss = (unorm_diff + magdiffc*3.0)?;
        let grad = loss.backward()?;
        let dw_vtx2xyz = grad.get(&vtx2xy).unwrap();
        let _ = vtx2xy.set(&vtx2xy.as_tensor().sub(&(dw_vtx2xyz * 0.05)?)?);
        if iter % 3 == 0 {
            let vtx2xy: Vec<_> = vtx2xy.flatten_all()?.to_vec1::<f32>()?;
            let hoge = del_msh::polyloop::to_cylinder_trimeshes(
                &vtx2xy, 2, 100.);
            // let _ = del_msh::io_obj::save_polyloop_(format!("target/polyloop_{}.obj", iter), &vtx2xy, 2);
            let _ = del_msh::io_obj::save_tri_mesh_(
                format!("target/polyloop_{}.obj", iter/3),
                &hoge.0, &hoge.1, 3);
        }
    }
    Ok(())
}