import numpy as np
from scipy import integrate
import scipy
from utils import list_of_dicts_to_dict_of_lists


class FlyWindDynamics:
    def __init__(self, polar_mode=False, control_mode='open_loop', update_every_step=True,
                 m=0.1, I=0.1, C_para=1.0, C_perp=1.0, C_phi=1.0):

        """ Initialize the fly-wind-dynamics simulator """

        # Set ODE solver parameter
        self.update_every_step = update_every_step

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

        # Initialize control gains
        self.Kp_para = 10.0  # proportional control constant for parallel speed
        self.Kp_perp = 0.0  # proportional control constant for perpendicular speed
        self.Kp_phi = 80.0  # proportional control constant for rotational speed
        self.Kd_phi = 3.0  # derivative control constant for rotational speed

        # Initialize the open-loop inputs
        self.t = 0.0  # current time
        self.u_para = np.array(0.0)  # parallel thrust
        self.u_perp = np.array(0.0)  # perpendicular thrust
        self.u_phi = np.array(0.0)  # rotational torque
        self.wdot = np.array(0.0)  # derivative of wind speed
        self.zetadot = np.array(0.0)  # derivative of wind angle in global frame

        # Current states & controls
        self.x = None  # state data
        self.u = None  # input data

        # Initialize the control commands (closed-loop reference values or open-loop)
        self.r_para = np.array(0.0)  # parallel thrust if open-loop or reference speed if closed-loop
        self.r_perp = np.array(0.0)  # perpendicular thrust if open-loop or reference speed if closed-loop
        self.r_phi = np.array(0.0)  # rotational torque if open-loop or reference rotational speed if closed-loop

        # Initialize variables to store simulation data
        self.t_solve = []
        self.x_solve = []
        self.dt = 0.0  # sample time
        self.xvel = 0.0  # x velocity in global frame
        self.yvel = 0.0  # y velocity in global frame
        self.xpos = 0.0  # x position in global frame
        self.ypos = 0.0  # x position in global frame
        self.sim_data = {}  # all simulation data in dictionary

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

    def update_inputs(self, x=None, t=None, r_para=0.0, r_perp=0.0, r_phi=0.0, wdot=0.0, zetadot=0.0):
        # Set state
        if x is None:
            x = self.x

        # Set time
        if t is not None:
            self.t = t

        # Set commands
        self.r_para = np.array(r_para)
        self.r_perp = np.array(r_perp)
        self.r_phi = np.array(r_phi)

        # Calculate control inputs
        self.u_para, self.u_perp, self.u_phi = self.calculate_control_inputs(r_para, r_perp, r_phi, x)

        # Set wind
        self.wdot = wdot
        self.zetadot = zetadot

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

    def set_controller_gains(self, Kp_para=10.0, Kp_perp=0.0, Kp_phi=80.0, Kd_phi=3.0):
        self.Kp_para = Kp_para  # proportional control constant for parallel speed
        self.Kp_perp = Kp_perp  # proportional control constant for perpendicular speed
        self.Kp_phi = Kp_phi  # proportional control constant for rotational speed
        self.Kd_phi = Kd_phi  # derivative control constant for rotational speed

    def calculate_control_inputs(self, r_para, r_perp, r_phi, states, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(states, flag2D=flag2D)

        # Calculate ground velocity & air velocity
        v_para, v_perp, g, psi = self.calculate_ground_velocity(states, flag2D=flag2D)
        a_para, a_perp, a, gamma = self.calculate_air_velocity(states, flag2D=flag2D)
        dir_of_travel = phi + psi

        # Calculate control input forces/torques based on control mode & control commands
        if self.control_mode == 'open_loop':
            u_para = np.array(r_para).copy()
            u_perp = np.array(r_perp).copy()
            u_phi = np.array(r_phi).copy()

        elif self.control_mode == 'align_psi':
            u_para = self.Kp_para * (r_para - v_para)
            u_perp = self.Kp_perp * (r_perp - v_perp)
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        elif self.control_mode == 'align_psi_constant_v_para':
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            u_perp = 0.0
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        elif self.control_mode == 'align_psi_constant_g':
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot) + (self.m * r_perp)
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        elif self.control_mode == 'align_phidot':
            u_para = self.Kp_para * (r_para - v_para)
            u_perp = self.Kp_perp * (r_perp - v_perp)
            u_phi = self.Kp_phi * (r_phi - phidot)

        elif self.control_mode == 'align_phidot_constant_v_para':
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            u_perp = 0.0
            u_phi = self.Kp_phi * (r_phi - phidot)

        elif self.control_mode == 'align_phidot_constant_g':
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot) + (self.m * r_perp)
            u_phi = self.Kp_phi * (r_phi - phidot)

        elif self.control_mode == 'hover':  # set thrust to cancel out wind, can add control afterwards
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot)
            u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot)
            u_phi = (self.C_phi * phidot)

        elif self.control_mode == 'no_dynamics_control':  # set thrust to cancel out wind, can add control afterwards
            a_para, a_perp, a, gamma = self.calculate_air_velocity(states, flag2D=flag2D)
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot) + (self.m * r_perp)
            u_phi = (self.C_phi * phidot) + (r_phi * self.I)

        elif self.control_mode == 'test':  # set thrust to cancel out wind, can add control afterwards
            u_para = (self.C_para * a_para) - (self.m * v_perp * phidot) + (self.m * r_para)
            # u_perp = (self.C_perp * a_perp) + (self.m * v_para * phidot) + (self.m * r_perp)
            u_perp = 0.0
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        else:
            raise Exception('control mode not available')

        return u_para, u_perp, u_phi  # return the open-loop control inputs

    def system_ode(self, t, x):
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

        self.t_solve.append(t)
        self.x_solve.append(x)

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x)

        # Compute drag
        D_para = self.C_para * a_para
        D_perp = self.C_perp * a_perp
        D_phi = self.C_phi * phidot

        # Update controls every time ODE solver is called
        if self.update_every_step:
            self.u_para, self.u_perp, self.u_phi \
                = self.calculate_control_inputs(self.r_para, self.r_perp, self.r_phi, x)

        # print(self.u_para)

        # Derivative of states
        xdot = np.array([((self.u_para - D_para) / self.m) + (v_perp * phidot),  # v_para_dot
                         ((self.u_perp - D_perp) / self.m) - (v_para * phidot),  # v_perp_dot
                         phidot,  # phidot
                         (self.u_phi / self.I) - (D_phi / self.I),  # phiddot
                         self.wdot,  # wdot
                         self.zetadot  # zetadot
                         ])

        return xdot

    def odeint_step(self, x0=None, dt=None, usim=None, polar=None, update_every_step=None):
        """ Solve ODE for one time step"""

        if polar is None:
            polar = self.polar_mode

        if update_every_step is not None:
            self.update_every_step = update_every_step

        if x0 is None:
            x0 = self.x.copy()
        else:
            x0 = x0.copy()

        if dt is None:
            dt = self.dt

        # Get inputs
        if usim is None:
            usim = np.zeros(self.n_input)

        usim = usim.copy()

        # Convert initial conditions & inputs to polar
        if polar:
            # First two states & inputs are polar
            g0 = x0[0]
            psi0 = x0[1]

            r_g = usim[0]
            r_psi = usim[1]

            # Convert polar states & controls to cartesian
            v_para_0, v_perp_0 = polar2cart(g0, psi0)

            r_para, r_perp = polar2cart(r_g, r_psi)

            # Replace initial polar states & inputs
            x0[0] = v_para_0
            x0[1] = v_perp_0

            usim[0] = r_para
            usim[1] = r_perp

        # Update control inputs & wind
        self.update_inputs(x0, r_para=usim[0], r_perp=usim[1], r_phi=usim[2], wdot=usim[3], zetadot=usim[4])

        # Integrate for one time step
        t_span = np.array([0, dt])
        x_solve = integrate.odeint(self.system_ode, x0, t_span, tcrit=t_span, tfirst=True)

        # print('t = ', np.round(self.t + dt, 5), ':', self.u_para, ':', len(self.t_solve))

        # Just get solution at t=dt
        self.x = x_solve[1]

        # Get states
        v_para, v_perp, phi, phidot, w, zeta = self.unpack_states(self.x, flag2D=True)

        # Calculate ground velocity
        v_para, v_perp, g, psi = self.calculate_ground_velocity(self.x, flag2D=True)

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(self.x, flag2D=True)
        dir_of_travel = phi + psi

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

        # Controls in polar coordinates
        u_g, u_psi = cart2polar(self.u_para, self.u_perp)

        # Update current time
        self.t = np.round(self.t + dt, 8)

        # Collect data
        data = {'time': self.t,

                'v_para': v_para,
                'v_perp': v_perp,
                'g': g,
                'psi': psi,

                'phi': phi,
                'phidot': phidot,

                'w': w,
                'zeta': zeta,
                'wdot': self.wdot,
                'zetadot': self.zetadot,

                'a_para': a_para,
                'a_perp': a_perp,
                'a': a,
                'gamma': gamma,
                'dir_of_travel': dir_of_travel,

                'xvel': xvel,
                'yvel': yvel,

                'xpos': self.xpos,
                'ypos': self.ypos,

                'u_para': self.u_para,
                'u_perp': self.u_perp,
                'u_phi': self.u_phi,
                'u_g': u_g,
                'u_psi': u_psi}

        # Output the state in polar coordinates if polar_mode is true
        self.x = self.x.copy()
        if polar:
            v_para = self.x[0]
            v_perp = self.x[1]
            g, psi = cart2polar(v_para, v_perp)
            self.x[0] = g
            self.x[1] = psi

        return self.x.copy(), data

    def odeint_simulate(self, x0, tsim, usim, polar=None, update_every_step=None):
        # Update ODE solver parameter
        if update_every_step is not None:
            self.update_every_step = update_every_step

        if polar is None:
            polar = self.polar_mode

        # Reset simulator
        self.reset()

        # Time step
        dt = np.mean(np.diff(tsim))

        # Run once at time 0 to get initial data
        x, data = self.odeint_step(x0=x0, dt=0.0, polar=polar)

        # Solve ODE in steps
        t_solve = [0]
        x_solve = [x]
        sim_data = [data]
        for n in range(1, tsim.shape[0]):  # for each data point in input time vector
            # Step
            x, data = self.odeint_step(x0=x, dt=dt, usim=usim[n, :], polar=polar)

            t_solve.append(self.t)
            x_solve.append(x)
            sim_data.append(data)

        # Concatenate state vectors & data
        t_solve = np.hstack(t_solve)
        x_solve = np.vstack(x_solve)
        sim_data = list_of_dicts_to_dict_of_lists(sim_data, make_array=True)

        return x_solve, sim_data, t_solve

    def reset(self, time=0.0, xpos=0.0, ypos=0.0, xvel=0.0, yvel=0.0):
        self.t = time
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.t_solve = []
        self.x_solve = []


def polar2cart(r, theta):
    # Transform polar to cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def cart2polar(x, y):
    # Transform cartesian to polar
    r = np.sqrt((x**2) + (y**2))
    theta = np.arctan2(y, x)

    return r, theta
