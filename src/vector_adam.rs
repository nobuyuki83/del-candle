use candle_core::{Tensor, Var};

pub struct ParamsAdam {
    /// Learning rate
    pub lr: f64,
    /// Coefficient for moving average of first moment
    pub beta_1: f64,
    /// Coefficient for moving average of second moment
    pub beta_2: f64,
    /// Term added to denominator to improve numerical stability
    pub eps: f64,
}

pub struct Optimizer {
    vtx2xyz: Var,
    m: Tensor,
    v: Tensor,
    params: ParamsAdam,
    ls:  del_ls::linearsystem::Solver<f32>,
    t: f64,
    pub tri2vtx: candle_core::Tensor,
    pub lambda: f64,
}

impl Optimizer {
    pub fn new(
        var: Var,
        learning_rate: f64,
        tri2vtx: candle_core::Tensor,
        num_vtx: usize,
        lambda: f64
    ) -> candle_core::Result<Self> {
        let ls = {
            let tri2vtx: Vec<usize> = tri2vtx.flatten_all()?.to_vec1::<u32>()?.iter().map(|v| *v as usize).collect();
            del_fem_core::laplace_tri3::to_linearsystem(&tri2vtx, num_vtx, 1., lambda as f32)
        };
        let adam_params = crate::vector_adam::ParamsAdam {
            lr: learning_rate,
            beta_1: 0.9,
            beta_2: 0.99,
            eps: 1.0e-12,
        };
        let dtype = var.dtype();
        assert!(dtype.is_float());
        let shape = var.shape();
        let m = Tensor::zeros(shape, dtype, &candle_core::Device::Cpu)?;
        let v = Tensor::zeros((shape.dims2()?.0,1), dtype, &candle_core::Device::Cpu)?;
        // let v = Var::zeros(shape, dtype, &candle_core::Device::Cpu)?;
        let adm = Optimizer {
            vtx2xyz: var,
            m,
            v,
            params: adam_params,
            t: 1.,
            ls,
            tri2vtx,
            lambda
        };
        Ok(adm)
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
        let b1 = self.params.beta_1;
        let b2 = self.params.beta_2;
        self.t += 1.;
        if let Some(dw_vtx2xyz) = grads.get(&self.vtx2xyz) {
            let num_vtx = dw_vtx2xyz.dims2()?.0;
            let grad = {
                self.ls.r_vec = dw_vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
                self.ls.solve_cg();
                Tensor::from_vec(self.ls.u_vec.clone(), (num_vtx, 3), &candle_core::Device::Cpu)?
            };
            self.m = ((b1 * self.m.clone())? + ((1. - b1) * grad.clone())?)?;
            let hoge = grad.clone().sqr()?.sum_keepdim(1)?;
            self.v = ((b2 * self.v.clone())? + ((1. - b2) * hoge )?)?;
            let m_hat = (&self.m / (1. - b1.powf(self.t)))?;
            let v_hat = (&self.v / (1. - b2.powf(self.t)))?;
            let delta = (m_hat * self.params.lr)?.broadcast_div(&(v_hat.sqrt()? + self.params.eps)?)?;
            // let delta = (grad * self.params.lr)?; // gradient descent
            let delta = {
                self.ls.r_vec = delta.flatten_all()?.to_vec1::<f32>()?;
                self.ls.solve_cg();
                Tensor::from_vec(self.ls.u_vec.clone(), (num_vtx, 3), &candle_core::Device::Cpu)?
            };
            self.vtx2xyz.set(&self.vtx2xyz.sub(&(delta))?)?;
        }
        Ok(())
    }
}

