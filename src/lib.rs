pub mod trimesh3_to_tri2nrm;
pub mod render_meshtri2_vtxcolor;
pub mod polyloop_to_edgevector;
pub mod polyloop2_to_area;
pub mod polyloop2_to_diffcoord;
pub mod site2_to_voronoi2;

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
