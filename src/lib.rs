mod trimesh3_to_tri2nrm;
mod meshtri2_vtxcolor_to_image;
mod polyloop_to_edgevector;
mod polyloop2_to_area;

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
