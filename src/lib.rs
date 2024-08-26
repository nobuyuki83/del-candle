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
    let (data, img_shape, depth) = del_canvas_core::load_image_as_float_array(&path);
    candle_core::Tensor::from_vec(
        data,
        candle_core::Shape::from((img_shape.0, img_shape.1, depth)),
        &candle_core::Device::Cpu,
    )
    .unwrap()
}
