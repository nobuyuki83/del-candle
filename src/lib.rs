pub mod cubic_stylization;
pub mod polygonmesh2_to_areas;
pub mod polygonmesh2_to_cogs;
pub mod polyloop2_to_diffcoord;
pub mod render_meshtri2_vtxcolor;
pub mod trimesh3_to_tri2nrm;
pub mod voronoi2;
pub mod vtx2xyz_to_edgevector;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
