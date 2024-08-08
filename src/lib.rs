pub mod bvh;
pub mod cubic_stylization;
pub mod diffcoord_polyloop2;
pub mod diffcoord_trimesh3;
pub mod gaussian_splatting;
pub mod gd_with_laplacian_reparam;
pub mod perturb_tensor;
pub mod polygonmesh2_to_areas;
pub mod polygonmesh2_to_cogs;
pub mod raycast_trimesh;
pub mod render_meshtri2_vtxcolor;
pub mod render_meshtri3_depth;
pub mod trimesh3_to_tri2nrm;
pub mod vector_adam;
pub mod voronoi2;
pub mod vtx2xyz_to_edgevector;

pub fn load_img_as_tensor<P>(path: P) -> candle_core::Tensor
where
    P: AsRef<std::path::Path>,
{
    use image::GenericImageView;
    let img_trg = image::open(path).unwrap();
    let (width, height) = img_trg.dimensions();
    let (width, height) = (width as usize, height as usize);
    let depth: usize = img_trg.color().bytes_per_pixel().into();
    let img_trg = img_trg.into_bytes();
    let img_trg: Vec<f32> = img_trg.iter().map(|&v| (v as f32) / 255.0f32).collect();
    assert_eq!(img_trg.len(), width * height * depth);
    candle_core::Tensor::from_vec(
        img_trg,
        candle_core::Shape::from((height, width, depth)),
        &candle_core::Device::Cpu,
    )
    .unwrap()
}
