import numpy as np
import torch
import biophys_cython_comp

solver = biophys_cython_comp.forwardeuler




def sample_box(size,low= torch.Tensor([0.5, 1e-4, 1e-4, 1e-4, 50.0, 40.0, 1e-4, 35.0]),
            high = torch.Tensor([80.0, 15.0, 0.6, 0.6, 3000.0, 90.0, 0.15, 100.0]),
            seed=None):
    #gbar_Na, gbar_K, g_leak, gbar_M, tau_max, Vt, nois_fact, E_leak


    return (
        torch.distributions.Independent(
            torch.distributions.Uniform(low=low, high=high),
            reinterpreted_batch_ndims=1,
        )
        .sample((size,))
    )

class HH_Biophys_Cython:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        self.seed = seed
        if seed is not None:
            biophys_cython_comp.seed(seed)
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, I, fineness=1, max_n_steps=float("inf")):
        """Simulates the model for a specified time duration."""

        biophys_cython_comp.setparams(self.params)
        tstep = float(dt)

        # explictly cast everything to double precision
        t = t.astype(np.float64)
        I = I.astype(np.float64)
        V = np.zeros_like(t).astype(np.float64)  # baseline voltage
        V[0] = float(self.state[0])
        n = np.zeros_like(t).astype(np.float64)
        m = np.zeros_like(t).astype(np.float64)
        h = np.zeros_like(t).astype(np.float64)
        p = np.zeros_like(t).astype(np.float64)
        q = np.zeros_like(t).astype(np.float64)
        r = np.zeros_like(t).astype(np.float64)
        u = np.zeros_like(t).astype(np.float64)
        r_mat = self.rng.randn(len(t)).astype(np.float64)

        solver(t, I, V, m, n, h, p, q, r, u, tstep, r_mat)

        return np.array(V).reshape(-1, 1)
    

class HH_Biophys:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, I):
        """Simulates the model for a specified time duration."""

        if len(self.params[0, :]) == 2:
            gbar_Na = self.params[0, 0]  # mS/cm2
            gbar_Na.astype(float)
            gbar_K = self.params[0, 1]
            gbar_K.astype(float)
            g_leak = 0.1
            gbar_M = 0.07
            tau_max = 6e2  # ms
            Vt = -60.0
            nois_fact = 0.1
            E_leak = -70.0
        else:
            gbar_Na = self.params[0, 0]  # mS/cm2
            gbar_Na.astype(float)
            gbar_K = self.params[0, 1]
            gbar_K.astype(float)
            g_leak = self.params[0, 2]
            g_leak.astype(float)
            gbar_M = self.params[0, 3]
            gbar_M.astype(float)
            # gbar_L = self.params[0,4]
            # gbar_L.astype(float)
            # gbar_T = self.params[0,5]
            # gbar_T.astype(float)
            tau_max = self.params[0, 4]  # ms
            tau_max.astype(float)
            Vt = -self.params[0, 5]
            Vt.astype(float)
            nois_fact = self.params[0, 6]
            nois_fact.astype(float)
            E_leak = -self.params[0, 7]
            E_leak.astype(float)

        tstep = float(dt)

        # Parameters
        nois_fact_obs = 0.0
        C = 1.0  # uF/cm2
        E_Na = 53  # mV
        E_K = -107

        # to generate burst
        E_Ca = 120
        Vx = 2
        gbar_L = 0
        gbar_T = 0

        ####################################
        # kinetics
        def Exp(z):
            if z < -5e2:
                return np.exp(-5e2)
            else:
                return np.exp(z)

        def efun(z):
            if np.abs(z) < 1e-4:
                return 1 - z / 2
            else:
                return z / (Exp(z) - 1)

        def alpha_m(x):
            v1 = x - Vt - 13.0
            return 0.32 * efun(-0.25 * v1) / 0.25

        def beta_m(x):
            v1 = x - Vt - 40
            return 0.28 * efun(0.2 * v1) / 0.2

        def alpha_h(x):
            v1 = x - Vt - 17.0
            return 0.128 * Exp(-v1 / 18.0)

        def beta_h(x):
            v1 = x - Vt - 40.0
            return 4.0 / (1 + Exp(-0.2 * v1))

        def alpha_n(x):
            v1 = x - Vt - 15.0
            return 0.032 * efun(-0.2 * v1) / 0.2

        def beta_n(x):
            v1 = x - Vt - 10.0
            return 0.5 * Exp(-v1 / 40)

        # slow non-inactivating K+
        def p_inf(x):
            v1 = x + 35.0
            return 1.0 / (1.0 + Exp(-0.1 * v1))

        def tau_p(x):
            v1 = x + 35.0
            return tau_max / (3.3 * Exp(0.05 * v1) + Exp(-0.05 * v1))

        # to generate burst
        # high-threshold Ca2+
        def alpha_q(x):
            v1 = x + 27
            return 0.055 * efun(-v1 / 3.8) * 3.8

        def beta_q(x):
            v1 = x + 75
            return 0.94 * Exp(-v1 / 17)

        def alpha_r(x):
            v1 = x + 13
            return 0.000457 * Exp(-v1 / 50)

        def beta_r(x):
            v1 = x + 15
            return 0.0065 / (1 + Exp(-v1 / 28))

        # low-threshold Ca2+
        def s_inf(x):
            v1 = x + Vx + 57
            return 1 / (1 + Exp(-v1 / 6.2))

        def u_inf(x):
            v1 = x + Vx + 81
            return 1 / (1 + Exp(v1 / 4))

        def tau_u(x):
            v1 = x + Vx + 84
            v2 = x + Vx + 113.2
            return 30.8 / 3.7 + (211.4 + Exp(v2 / 5)) / (3.7 * (1 + Exp(v1 / 3.2)))

        def tau_n(x):
            return 1 / (alpha_n(x) + beta_n(x))

        def n_inf(x):
            return alpha_n(x) / (alpha_n(x) + beta_n(x))

        def tau_m(x):
            return 1 / (alpha_m(x) + beta_m(x))

        def m_inf(x):
            return alpha_m(x) / (alpha_m(x) + beta_m(x))

        def tau_h(x):
            return 1 / (alpha_h(x) + beta_h(x))

        def h_inf(x):
            return alpha_h(x) / (alpha_h(x) + beta_h(x))

        def tau_q(x):
            return 1 / (alpha_q(x) + beta_q(x))

        def q_inf(x):
            return alpha_q(x) / (alpha_q(x) + beta_q(x))

        def tau_r(x):
            return 1 / (alpha_r(x) + beta_r(x))

        def r_inf(x):
            return alpha_r(x) / (alpha_r(x) + beta_r(x))

        ####################################

        # simulation from initial point
        V = np.zeros_like(t)  # baseline voltage
        n = np.zeros_like(t)
        m = np.zeros_like(t)
        h = np.zeros_like(t)
        p = np.zeros_like(t)
        q = np.zeros_like(t)
        r = np.zeros_like(t)
        u = np.zeros_like(t)

        V[0] = float(self.state[0])
        # V[0] = E_leak
        n[0] = n_inf(V[0])
        m[0] = m_inf(V[0])
        h[0] = h_inf(V[0])
        p[0] = p_inf(V[0])
        q[0] = q_inf(V[0])
        r[0] = r_inf(V[0])
        u[0] = u_inf(V[0])

        for i in range(1, t.shape[0]):
            tau_V_inv = (
                (m[i - 1] ** 3) * gbar_Na * h[i - 1]
                + (n[i - 1] ** 4) * gbar_K
                + g_leak
                + gbar_M * p[i - 1]
                + gbar_L * (q[i - 1] ** 2) * r[i - 1]
                + gbar_T * (s_inf(V[i - 1]) ** 2) * u[i - 1]
            ) / C
            V_inf = (
                (m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na
                + (n[i - 1] ** 4) * gbar_K * E_K
                + g_leak * E_leak
                + gbar_M * p[i - 1] * E_K
                + gbar_L * (q[i - 1] ** 2) * r[i - 1] * E_Ca
                + gbar_T * (s_inf(V[i - 1]) ** 2) * u[i - 1] * E_Ca
                + I[i - 1]
                + nois_fact * self.rng.randn() / (tstep**0.5)
            ) / (tau_V_inv * C)
            V[i] = V_inf + (V[i - 1] - V_inf) * Exp(-tstep * tau_V_inv)
            n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * Exp(-tstep / tau_n(V[i]))
            m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * Exp(-tstep / tau_m(V[i]))
            h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * Exp(-tstep / tau_h(V[i]))
            p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * Exp(-tstep / tau_p(V[i]))
            q[i] = q_inf(V[i]) + (q[i - 1] - q_inf(V[i])) * Exp(-tstep / tau_q(V[i]))
            r[i] = r_inf(V[i]) + (r[i - 1] - r_inf(V[i])) * Exp(-tstep / tau_r(V[i]))
            u[i] = u_inf(V[i]) + (u[i - 1] - u_inf(V[i])) * Exp(-tstep / tau_u(V[i]))

        #        return np.array(V).reshape(-1,1)
        return np.array(V).reshape(-1, 1) + nois_fact_obs * self.rng.randn(
            t.shape[0], 1
        )

def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x


def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x


class HodgkinHuxley:
    def __init__(self, I, dt, V0, cython=False, prior_log=False, reduced_model=False):
        """Hodgkin-Huxley simulator
        Parameters
        ----------
        I : array
            Numpy array with the input I
        dt : float
            Timestep
        V0 : float
            Voltage at first time step
        cython : bool
            If True, will use cython version of simulator (different import)
        reduced_model : bool
            If True, model with 2 parameters instead of 8
        seed : int or None
            If set, randomness across runs is disabled
        """
        if reduced_model:
            dim_param = 2
        else:
            dim_param = 8

        self.I = I
        self.cython = cython
        self.dt = dt
        self.t = np.arange(0, len(self.I), 1) * self.dt
        self.prior_log = prior_log

        # parameters that globally govern the simulations
        self.init = [V0]  # =V0

    def gen_single(self, params, seed):
        """Forward model for simulator for single parameter set
        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector
        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = param_invtransform(self.prior_log, np.asarray(params))

        assert params.ndim == 1, "params.ndim must be 1"
        if self.cython:
            hh = HH_Biophys_Cython(self.init, params.reshape(1, -1), seed=seed)
        else:
            hh = HH_Biophys(self.init, params.reshape(1, -1), seed=seed)
        states = hh.sim_time(self.dt, self.t, self.I)

        return {
            "data": states.reshape(-1),
            "time": self.t,
            "dt": self.dt,
            "I": self.I.reshape(-1),
        }