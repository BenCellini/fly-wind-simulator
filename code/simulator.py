import numpy as np
from scipy import integrate
import scipy


class FlyWindDynamics:
    def __init__(self, polar_mode=False, control_mode='open_loop',
                 m=0.1, I=0.1, C_para=1.0, C_perp=1.0, C_phi=1.0):

        """ Initialize the fly-wind-dynamics simulator """

        # Set state names
        #   v_para: parallel speed in fly frame
        #   v_perp: perpendicular speed in fly frame
        #   phi: heading angle in global frame
        #   phidot: angular velocity
        #   w: wind speed
        #   zeta: wind angle in global frame
        self.state_names = ['v_para', 'v_perp', 'phi', 'phidot', 'w', 'zeta']  # state names

        # Set input names
        #   u_para: parallel thrust in fly frame
        #   u_perp: perpendicular thrust in fly frame
        #   u_phi: rotational torque
        #   wdot: derivative of wind speed
        #   zetadot: derivative of wind angle
        self.input_names = ['u_para', 'u_perp', 'u_phi', 'wdot', 'zetadot']  # input names

        # If polar mode is on, change first two states to ground speed (g) & ground speed direction (psi) in fly frame
        # and change inputs to thrust magnitude (u_g) & direction (u_psi)
        self.polar_mode = polar_mode
        if self.polar_mode:
            self.state_names[0] = 'g'
            self.state_names[1] = 'psi'

            self.input_names[0] = 'u_g'
            self.input_names[1] = 'u_psi'

        # Set sizes
        self.n_state = len(self.state_names)  # number of states
        self.n_input = len(self.input_names)  # number of inputs

        # Set parameter values
        self.m = m  # mass
        self.I = I  # inertia
        self.C_para = C_para  # parallel damping
        self.C_perp = C_perp  # perpendicular damping
        self.C_phi = C_phi  # rotational damping

        # Initialize controller mode
        #   'open-loop': directly set the inputs
        #   'hover': set the inputs such that the simulated fly doesn't move no matter what wind does
        #   'no_dynamics': all dynamics are canceled out automatically by inputs, directly set trajectory
        #   'align_phi': PD controller to control heading
        self.control_mode = control_mode

        # Initialize the open-loop control inputs
        self.u_para = None  # parallel thrust
        self.u_perp = None  # perpendicular thrust
        self.u_phi = None  # rotational torque

        # Initialize the control commands (closed-loop reference values or open-loop)
        self.tsim = None  # time vector
        self.r_para = None  # parallel thrust if open-loop or reference speed if closed-loop
        self.r_perp = None  # perpendicular thrust if open-loop or reference speed if closed-loop
        self.r_phi = None  # rotational torque if open-loop or reference rotational speed if closed-loop
        self.wdot = None  # derivative of wind speed
        self.zetadot = None  # derivative of wind angle in global frame

        # Initialize interpolaters for controls
        self.interpolate = {}  # dictionary of interpolaters for each control command

        # Define the control commands for step ODE solver
        self.r_para_step = 0.0
        self.r_perp_step = 0.0
        self.r_phi_step = 0.0
        self.wdot_step = 0.0
        self.zetadot_step = 0.0

        # Initialize variables to store simulation data
        self.dt = 0.0  # sample time
        self.t = 0.0  # current time
        self.x = None  # state data
        self.u = None  # input data
        self.xvel = 0.0  # x velocity in global frame
        self.yvel = 0.0  # y velocity in global frame
        self.xpos = 0.0  # x position in global frame
        self.ypos = 0.0  # x position in global frame
        self.sim_data = {}  # all simulation data in dictionary

        # Initialize ODE solver
        self.solver = integrate.ode(self.system_ode).set_integrator('vode', method='bdf')

    def unpack_states(self, x, flag2D=False):
        if not flag2D:
            v_para, v_perp, phi, phidot, w, zeta = x
        else:
            x = np.atleast_2d(x)
            v_para = x[:, 0]
            v_perp = x[:, 1]
            phi = x[:, 2]
            phidot = x[:, 3]
            w = x[:, 4]
            zeta = x[:, 5]

        return v_para, v_perp, phi, phidot, w, zeta

    def set_control_commands(self, tsim, r_para=None, r_perp=None, r_phi=None, wdot=None, zetadot=None):
        # Time
        self.tsim = tsim
        n_point = self.tsim.shape[0]

        # Set commands to 0 if not given
        if r_para is None:
            self.r_para = np.zeros((n_point, 1))
        else:
            self.r_para = r_para.copy()

        if r_perp is None:
            self.r_perp = np.zeros((n_point, 1))
        else:
            self.r_perp = r_perp.copy()

        if r_phi is None:
            self.r_phi = np.zeros((n_point, 1))
        else:
            self.r_phi = r_phi.copy()

        if wdot is None:
            self.wdot = np.zeros((n_point, 1))
        else:
            self.wdot = wdot.copy()

        if zetadot is None:
            self.zetadot = np.zeros((n_point, 1))
        else:
            self.zetadot = zetadot.copy()

    def calculate_air_velocity(self, states, flag2D=False, w_direct=None):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(states, flag2D=flag2D)

        # If w is set directly
        if w_direct is not None:
            w = w_direct.copy()

        # Air speed in parallel & perpendicular directions
        a_para = v_para - w * np.cos(phi - zeta)
        a_perp = v_perp + w * np.sin(phi - zeta)

        # Air velocity angle & magnitude
        a = np.linalg.norm((a_perp, a_para), ord=2, axis=0)  # air velocity magnitude
        gamma = np.arctan2(a_perp, a_para)  # air velocity angle

        return a_para, a_perp, a, gamma

    def calculate_ground_velocity(self, states, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(states, flag2D=flag2D)

        # Ground velocity angle & magnitude
        g = np.linalg.norm((v_perp, v_para), ord=2, axis=0)  # ground velocity magnitude
        psi = np.arctan2(v_perp, v_para)  # ground velocity angle

        return v_para, v_perp, g, psi

    def calculate_dir_of_travel(self, states, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(states, flag2D=flag2D)

        # Ground velocity angle
        v_para, v_perp, g, psi = self.calculate_ground_velocity(states, flag2D=flag2D)

        # Get heading velocity, air velocity, & direction-of-travel angles
        dir_of_travel = psi + phi  # direction of travel in global frame

        return dir_of_travel

    def calculate_control_inputs(self, r_para, r_perp, r_phi, states, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(states, flag2D=flag2D)

        # Calculate control input forces/torques based on control mode & control commands
        if self.control_mode == 'open_loop':
            u_para = r_para.copy()
            u_perp = r_perp.copy()
            u_phi = r_phi.copy()

        elif self.control_mode == 'align_phi':
            u_para = r_para.copy()
            u_perp = r_perp.copy()
            u_phi = r_phi.copy()

        elif self.control_mode == 'hover':  # set thrust to cancel out wind, can add control afterwards
            a_para, a_perp, a, gamma = self.calculate_air_velocity(states, flag2D=flag2D)
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot)
            u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot)
            u_phi = (self.C_phi * phidot)

        elif self.control_mode == 'no_dynamics_control':  # set thrust to cancel out wind, can add control afterwards
            a_para, a_perp, a, gamma = self.calculate_air_velocity(states, flag2D=flag2D)
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot) + (self.m * r_perp)
            u_phi = (self.C_phi * phidot) + (r_phi * self.I)

        else:
            raise Exception("'control_mode' must be set to 'open_loop', 'hover', or 'no_dynamics'")

        return u_para, u_perp, u_phi  # return the open-loop control inputs

    def system_ode(self, t, x, mode):
        """ Dynamical system model.

        Inputs
            x: current states (tuple)
                v_para - parallel speed
                v_perp - perpendicular speed
                phi - orientation in global frame
                phidot - change in orientation angle
                w - wind speed
                zeta - wind angle in global frame
            t: current time

        Outputs
            xdot: derivative of states
        """

        # Get states
        v_para, v_perp, phi, phidot, w, zeta = x

        # Get current control commands
        if mode == 'step':  # step mode
            r_para = np.array(self.r_para_step)
            r_perp = np.array(self.r_perp_step)
            r_phi = np.array(self.r_phi_step)
            wdot = np.array(self.wdot_step)
            zetadot = np.array(self.zetadot_step)

        elif mode == 'series':  # for preset time series
            r_para = self.interpolate['r_para'](t)
            r_perp = self.interpolate['r_perp'](t)
            r_phi = self.interpolate['r_phi'](t)
            wdot = self.interpolate['wdot'](t)
            zetadot = self.interpolate['zetadot'](t)

        else:
            raise Exception('ODE mode must be "step" or "series"')

        # Calculate control inputs
        u_para, u_perp, u_phi = self.calculate_control_inputs(r_para, r_perp, r_phi, x)

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x)

        # Compute drag
        D_para = self.C_para * a_para
        D_perp = self.C_perp * a_perp
        D_phi = self.C_phi * phidot

        # Derivative of states
        xdot = np.array([((u_para - D_para) / self.m) + (v_perp * phidot),  # v_para_dot
                         ((u_perp - D_perp) / self.m) - (v_para * phidot),  # v_perp_dot
                         phidot,  # phidot
                         (u_phi / self.I) - (D_phi / self.I),  # phiddot
                         wdot,  # wdot
                         zetadot  # zetadot
                         ])

        return xdot

    def odeint_step(self, x0, dt=None, polar=None,
                    r_g=0.0, r_psi=0.0, r_para=0.0, r_perp=0.0, r_phi=0.0, wdot=0.0, zetadot=0.0):

        x0 = x0.copy()

        if dt is None:
            dt = self.dt

        if polar is None:
            polar = self.polar_mode

        if polar:
            # First two states are polar
            g0 = x0[0]
            psi0 = x0[1]

            # Convert polar states & controls to cartesian
            v_para_0, v_perp_0, r_para, r_perp = self.convert_to_cartesian(g0, psi0, r_g, r_psi)

            # Replace initial polar states
            x0[0] = v_para_0
            x0[1] = v_perp_0

        # Set controls
        self.r_para_step = r_para
        self.r_perp_step = r_perp
        self.r_phi_step = r_phi
        self.wdot_step = wdot
        self.zetadot_step = zetadot

        # Integrate for one time step
        t_span = np.array([0, dt])
        x_solve = integrate.odeint(self.system_ode, x0, t_span,
                                   tcrit=t_span, tfirst=True, args=('step',))

        # Just get solution at t=dt
        x_step = x_solve[1]

        # print(r_phi, ':', x_step)

        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(x_step, flag2D=True)

        # Calculate ground velocity
        v_para, v_perp, g, psi = self.calculate_ground_velocity(x_step, flag2D=True)

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x_step, flag2D=True)

        # Ground velocity in global frame
        xvel = v_para * np.cos(phi) - v_perp * np.sin(phi)
        yvel = v_para * np.sin(phi) + v_perp * np.cos(phi)

        # Compute change in position
        xvel_v = np.hstack((self.xvel, xvel))
        yvel_y = np.hstack((self.yvel, yvel))

        delta_xpos = scipy.integrate.trapz(xvel_v, dx=dt)
        delta_ypos = scipy.integrate.trapz(yvel_y, dx=dt)

        # New position
        self.xpos = self.xpos + delta_xpos
        self.ypos = self.ypos + delta_ypos

        # Update current velocity
        self.xvel = xvel.copy()
        self.yvel = yvel.copy()

        # Update current time
        self.t = self.t + dt

        # Collect data
        data = {'time': self.t,
                'v_para': v_para,
                'v_perp': v_perp,
                'phi': phi,
                'phidot': phidot,
                'w': w,
                'zeta': zeta,
                'g': g,
                'psi': psi,
                'a': a,
                'gamma': gamma,
                'xvel': xvel,
                'yvel': yvel,
                'xpos': self.xpos,
                'ypos': self.ypos}

        return x_step, data

    def odeint_simulate(self, x0, tsim, usim):
        # Set the control commands
        self.set_control_commands(tsim,
                                  r_para=usim[:, 0],
                                  r_perp=usim[:, 1],
                                  r_phi=usim[:, 2],
                                  wdot=usim[:, 3],
                                  zetadot=usim[:, 4])

        # Time step
        dt = np.mean(np.diff(tsim))

        # Solve ODE
        t_solve = [0]
        x_solve = [x0]
        self.reset()
        for n in range(1, tsim.shape[0]):
            # print(self.r_phi[n], ':', x_solve[-1])
            x_step, data = self.odeint_step(x_solve[-1], dt=dt, polar=False,
                                            r_para=self.r_para[n],
                                            r_perp=self.r_perp[n],
                                            r_phi=self.r_phi[n],
                                            wdot=self.wdot[n],
                                            zetadot=self.zetadot[n])

            t_solve.append(t_solve[-1] + dt)
            x_solve.append(x_step)

        # Concatenate state vectors
        x_solve = np.vstack(x_solve)

        return x_solve, t_solve

    def reset(self, time=0.0, xpos=0.0, ypos=0.0, xvel=0.0, yvel=0.0):
        self.t = time
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel

    def convert_to_cartesian(self, g0, psi0, r_g, r_psi):
        # Transform polar initial condition to cartesian
        v_para_0 = g0 * np.cos(psi0)  # parallel ground speed
        v_perp_0 = g0 * np.sin(psi0)  # perpendicular ground speed

        # Transform & replace control inputs
        r_para = r_g * np.cos(r_psi)  # parallel thrust
        r_perp = r_g * np.sin(r_psi)  # perpendicular thrust

        return v_para_0, v_perp_0, r_para, r_perp

    def simulate(self, x0, tsim, usim):
        """ Simulate dynamics.

        Inputs
            x0: initial state vector
            tsim: time vector
            usim: inputs as columns
                1st - parallel thrust (thrust magnitude if in polar mode)
                2nd - perpendicular thrust (thrust direction if in polar mode)
                3rd - rotational torque
                4th - derivative of wind speed
                5th - derivative of wind angle in global frame

        Outputs
            sim_data: dictionary containing simulation data
        """

        # Sampling time
        self.dt = np.mean(np.diff(tsim))

        if self.polar_mode:
            # Replace state & input names
            self.state_names[0] = 'g'
            self.state_names[1] = 'psi'

            self.input_names[0] = 'u_g'
            self.input_names[1] = 'u_psi'

            # Transform polar initial condition to cartesian
            g0 = x0[0]  # ground speed
            psi0 = x0[1]  # ground speed angle
            v_para_0 = g0 * np.cos(psi0)  # parallel ground speed
            v_perp_0 = g0 * np.sin(psi0)  # perpendicular ground speed

            # Replace initial conditions with model states
            x0[0] = v_para_0
            x0[1] = v_perp_0

            # Transform & replace control inputs
            r_g = usim[:, 0].copy()  # forward thrust magnitude
            r_psi = usim[:, 1].copy()  # forward thrust angle
            r_para = r_g * np.cos(r_psi)  # parallel thrust
            r_perp = r_g * np.sin(r_psi)  # perpendicular thrust

            usim = usim.copy()
            usim[:, 0] = r_para
            usim[:, 1] = r_perp

        # Set the control commands
        self.set_control_commands(tsim,
                                  r_para=usim[:, 0],
                                  r_perp=usim[:, 1],
                                  r_phi=usim[:, 2],
                                  wdot=usim[:, 3],
                                  zetadot=usim[:, 4])

        # Define interpolaters for each control input
        interp_method = 'nearest'
        bounds_error = False
        self.interpolate['r_para'] = scipy.interpolate.interp1d(self.tsim, self.r_para, kind=interp_method,
                                                                fill_value=self.r_para[-1], bounds_error=bounds_error)
        self.interpolate['r_perp'] = scipy.interpolate.interp1d(self.tsim, self.r_perp, kind=interp_method,
                                                                fill_value=self.r_perp[-1], bounds_error=bounds_error)
        self.interpolate['r_phi'] = scipy.interpolate.interp1d(self.tsim, self.r_phi, kind=interp_method,
                                                               fill_value=self.r_phi[-1], bounds_error=bounds_error)
        self.interpolate['wdot'] = scipy.interpolate.interp1d(self.tsim, self.wdot, kind=interp_method,
                                                              fill_value=self.wdot[-1], bounds_error=bounds_error)
        self.interpolate['zetadot'] = scipy.interpolate.interp1d(self.tsim, self.zetadot, kind=interp_method,
                                                                 fill_value=self.r_para[-1], bounds_error=bounds_error)

        # Simulate the system & calculate the states over time
        # x = integrate.odeint(self.system_ode, x0, tsim, tcrit=tsim, tfirst=True, args=('series',))
        # # x, t_solve = self.ode_simulate(x0, tsim, usim)
        x, t_solve = self.odeint_simulate(x0, tsim, usim)

        # Store state vectors
        self.x = x

        # Get the states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(x, flag2D=True)

        # Ground velocity
        v_para, v_perp, g, psi = self.calculate_ground_velocity(x, flag2D=True)

        # Air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x, flag2D=True)

        # Relative angle between air velocity & wind velocity
        beta = phi - zeta

        # Direction of travel
        dir_of_travel = self.calculate_dir_of_travel(x, flag2D=True)

        # Ground velocity in global frame
        xvel = v_para * np.cos(phi) - v_perp * np.sin(phi)
        yvel = v_para * np.sin(phi) + v_perp * np.cos(phi)

        # Position
        xpos = integrate.cumtrapz(xvel, tsim, initial=0)
        ypos = integrate.cumtrapz(yvel, tsim, initial=0)

        # Control inputs from control commands
        u_para, u_perp, u_phi = self.calculate_control_inputs(self.r_para, self.r_perp, self.r_phi, x, flag2D=True)
        self.u = np.stack((u_para, u_perp, u_phi, self.wdot, self.zetadot), axis=1)

        # Thrust magnitude & direction
        u_g = np.linalg.norm((u_para, u_perp), ord=2, axis=0)  # thrust magnitude
        u_psi = np.arctan2(u_perp, u_para)  # thrust angle

        # Drag
        D_para = self.C_para * a_para
        D_perp = self.C_perp * a_perp
        D_phi = self.C_phi * phidot

        # Acceleration
        v_para_dot = ((u_para - D_para) / self.m) + (v_perp * phidot)  # parallel acceleration
        v_perp_dot = ((u_perp - D_perp) / self.m) - (v_para * phidot)  # perpendicular acceleration
        q = np.linalg.norm((v_para_dot, v_perp_dot), ord=2, axis=0)  # acceleration magnitude
        alpha = (np.arctan2(v_perp_dot, v_para_dot))  # acceleration angle

        # Angular acceleration
        phiddot = (u_phi / self.I) - (D_phi / self.I)

        # Output dictionary
        sim_data = {'time': tsim,
                    'x': x,
                    'v_para': v_para,
                    'v_perp': v_perp,
                    'phi': phi,
                    'phidot': phidot,
                    'w': w,
                    'zeta': zeta,
                    'xvel': xvel,
                    'yvel': yvel,
                    'xpos': xpos,
                    'ypos': ypos,
                    'g': g,
                    'psi': psi,
                    'a': a,
                    'gamma': gamma,
                    'a_para': a_para,
                    'a_perp': a_perp,
                    'q': q,
                    'alpha': alpha,
                    'beta': beta,
                    'dir_of_travel': dir_of_travel,
                    'r_para': self.r_para,
                    'r_perp': self.r_perp,
                    'r_phi': self.r_phi,
                    'u_para': u_para,
                    'u_perp': u_perp,
                    'u_phi': u_phi,
                    'u_g': u_g,
                    'u_psi': u_psi,
                    'v_para_dot': v_para_dot,
                    'v_perp_dot': v_perp_dot,
                    'phiddot': phiddot,
                    'wdot': self.wdot,
                    'zetadot': self.zetadot,
                    }

        return sim_data
