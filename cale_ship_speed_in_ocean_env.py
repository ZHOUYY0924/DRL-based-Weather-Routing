import math
import numpy as np
import scipy.interpolate as spi

def get_Lap_and_Keller_Chart():
    x_data = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.67])
    y_data = np.array([22, 22, 23, 26, 33, 50, 60])
    spl = spi.splrep(x_data, y_data, k=3)

    return spl

def get_Rt(V):
    spl = get_Lap_and_Keller_Chart()

    rho = 1025.91
    displ = 312_600  # Displacement in [m³]
    C_b = 0.810 # Block Coeffiient
    L_w = 320
    B = 58
    dm = 20.8
    Am = B * dm
    Cp = 0.83
    viscosity_coefficient = 1.18831 * 10**6
    Froude_number = 0.142
    Reynold_number = V * L_w / viscosity_coefficient

    S = 27194

    Cf = (0.4631 / math.log(V * L_w / Reynold_number, 10) ** 2.6)
    Cr = spi.splev(V/math.sqrt(Cp*L_w), spl) * Am / S * 10**(-3)
    delta_Car = -0.0001

    Ct = (Cf + Cr + delta_Car)
    delta_Ct = Ct * (B/dm-2.4) * 0.5/100
    Ct = Ct + delta_Ct

    Rt = 1/2 * rho * S * Ct * (V**2)
    return Rt

def get_V(Rt):
    V_list = np.arange(0.1, 16.1, 0.1).tolist()
    Rt_list = [get_Rt(each_V) for each_V in V_list]
    V_array = np.array(V_list)
    Rt_array = np.array(Rt_list)
    spl = spi.splrep(Rt_array, V_array, k=3)
    V = spi.splev(Rt, spl)

    return V

def Calculate_wind_speed_and_driection(u,v):
    wind_speed = math.sqrt(u ** 2 + v ** 2)
    tmp = 270.0 - math.atan2(v, u) * 180.0 / math.pi
    wind_driection = math.fmod(tmp, 360.0)
    return wind_speed,round(wind_driection,2)


def calculate_ship_speed_u_v(wind_speed, wind_direction):
    wind_direction_rad = np.radians(wind_direction)
    U10 = wind_speed * np.sin(wind_direction_rad)
    V10 = wind_speed * np.cos(wind_direction_rad)

    return U10, V10

def Force_and_moment_of_wind(wind_speed, wind_driection,SOG,heading):
    if heading < 0:
        heading = heading + math.pi

    def sgn(x):
        if x < 0:
            return -1
        elif x == 0:
            return 0
        else:
            return 1

    density_of_air = 1.224
    Af = 1200
    As = 3600
    Loa = 320
    B = 58
    M = 1
    Ass = 807.71
    c = 5 * 360
    e = 17.4 + 325.5/2

    u, v = calculate_ship_speed_u_v(SOG,heading)
    Ur = -u + wind_speed * math.cos(math.radians(wind_driection) - heading)
    Vr = -v + wind_speed * math.sin(math.radians(wind_driection) - heading)

    alpha_R = np.arctan2(Vr, Ur)
    if Ur > 0 and Vr > 0:
        alpha_R = alpha_R
    elif Ur > 0 and Vr < 0:
        alpha_R = np.pi + alpha_R
    elif Ur < 0 and Vr > 0:
        alpha_R = -alpha_R
    elif Ur < 0 and Vr < 0:
        alpha_R = np.pi + alpha_R

    alpha_R_degree = round(math.degrees(alpha_R), 5)
    SOG = math.sqrt(u ** 2 + v ** 2)
    UR = math.sqrt(wind_speed ** 2 + SOG ** 2 + 2 * wind_speed * SOG * math.cos(math.radians(wind_driection)- math.atan(v / u)))

    # angle    A_0     A_1       A_2      A_3      A_4      A_5     A_6
    surge_coefficients = np.array( \
        [[0.0, 2.1520, -5.000, 0.2430, -0.1640, 0.0000, 0.000, 0.000], \
         [10.0, 1.7140, -3.330, 0.1450, -0.1210, 0.0000, 0.000, 0.000], \
         [20.0, 1.8180, -3.970, 0.2110, -0.1430, 0.0000, 0.000, 0.033], \
         [30.0, 1.9650, -4.810, 0.2430, -0.1540, 0.0000, 0.000, 0.041], \
         [40.0, 2.3330, -5.990, 0.2470, -0.1900, 0.0000, 0.000, 0.042], \
         [50.0, 1.7260, -6.540, 0.1890, -0.1730, 0.3480, 0.000, 0.048], \
         [60.0, 0.9130, -4.680, 0.0000, -0.1040, 0.4820, 0.000, 0.052], \
         [70.0, 0.4570, -2.880, 0.0000, -0.0680, 0.3460, 0.000, 0.043], \
         [80.0, 0.3410, -0.910, 0.0000, -0.0310, 0.0000, 0.000, 0.032], \
         [90.0, 0.3550, 0.000, 0.0000, 0.0000, -0.2470, 0.000, 0.018], \
         [100.0, 0.6010, 0.000, 0.0000, 0.0000, -0.3720, 0.000, -0.020], \
         [110.0, 0.6510, 1.290, 0.0000, 0.0000, -0.5820, 0.000, -0.031], \
         [120.0, 0.5640, 2.540, 0.0000, 0.0000, -0.7480, 0.000, -0.024], \
         [130.0, -0.1420, 3.580, 0.0000, 0.0470, -0.7000, 0.000, -0.028], \
         [140.0, -0.6770, 3.640, 0.0000, 0.0690, -0.5290, 0.000, -0.032], \
         [150.0, -0.7230, 3.140, 0.0000, 0.0640, -0.4750, 0.000, -0.032], \
         [160.0, -2.1480, 2.560, 0.0000, 0.0810, 0.0000, 1.270, -0.027], \
         [170.0, -2.7070, 3.970, -0.1750, 0.1260, 0.0000, 1.810, 0.000], \
         [180.0, -2.5290, 3.760, -0.1740, 0.1280, 0.0000, 1.550, 0.000]])

    # angle    B_0     B_1       B_2      B_3      B_4      B_5     B_6
    sway_coefficients = np.array( \
        [[0.0, 0.0000, 0.000, 0.0000, 0.0000, 0.0000, 0.000, 0.000], \
         [10.0, 0.0960, 0.220, 0.0000, 0.0000, 0.0000, 0.000, 0.000], \
         [20.0, 0.1760, 0.710, 0.0000, 0.0000, 0.0000, 0.000, 0.000], \
         [30.0, 0.2250, 1.380, 0.0000, 0.0230, 0.0000, -0.290, 0.000], \
         [40.0, 0.3290, 1.820, 0.0000, 0.0430, 0.0000, -0.590, 0.000], \
         [50.0, 1.1640, 1.260, 0.1210, 0.0000, -0.2420, -0.950, 0.000], \
         [60.0, 1.1630, 0.960, 0.1010, 0.0000, -0.1770, -0.880, 0.000], \
         [70.0, 0.9160, 0.530, 0.0690, 0.0000, 0.0000, -0.650, 0.000], \
         [80.0, 0.8440, 0.550, 0.0820, 0.0000, 0.0000, -0.540, 0.000], \
         [90.0, 0.8890, 0.000, 0.1380, 0.0000, 0.0000, -0.660, 0.000], \
         [100.0, 0.7990, 0.000, 0.1550, 0.0000, 0.0000, -0.550, 0.000], \
         [110.0, 0.7970, 0.000, 0.1510, 0.0000, 0.0000, -0.550, 0.000], \
         [120.0, 0.9960, 0.000, 0.1840, 0.0000, -0.2120, -0.660, 0.340], \
         [130.0, 1.0140, 0.000, 0.1910, 0.0000, -0.2800, -0.690, 0.440], \
         [140.0, 0.7840, 0.000, 0.1660, 0.0000, -0.2090, -0.530, 0.380], \
         [150.0, 0.5360, 0.000, 0.1760, -0.0290, -0.1630, 0.000, 0.270], \
         [160.0, 0.2510, 0.000, 0.1060, -0.0220, 0.0000, 0.000, 0.000], \
         [170.0, 0.1250, 0.000, 0.0460, -0.0120, 0.0000, 0.000, 0.000], \
         [180.0, 0.0000, 0.000, 0.0000, 0.0000, 0.0000, 0.000, 0.000]])

    # angle    C_0     C_1       C_2      C_3      C_4      C_5
    yaw_coefficients = np.array( \
        [[0.0, 0.0000, 0.000, 0.0000, 0.0000, 0.0000, 0.000], \
         [10.0, 0.0596, 0.061, 0.0000, 0.0000, 0.0000, -0.074], \
         [20.0, 0.1106, 0.204, 0.0000, 0.0000, 0.0000, -0.170], \
         [30.0, 0.2258, 0.245, 0.0000, 0.0000, 0.0000, -0.380], \
         [40.0, 0.2017, 0.457, 0.0000, 0.0067, 0.0000, -0.472], \
         [50.0, 0.1759, 0.573, 0.0000, 0.0118, 0.0000, -0.523], \
         [60.0, 0.1925, 0.480, 0.0000, 0.0115, 0.0000, -0.546], \
         [70.0, 0.2133, 0.315, 0.0000, 0.0081, 0.0000, -0.526], \
         [80.0, 0.1827, 0.254, 0.0000, 0.0053, 0.0000, -0.443], \
         [90.0, 0.2627, 0.000, 0.0000, 0.0000, 0.0000, -0.508], \
         [100.0, 0.2102, 0.000, -0.0195, 0.0000, 0.0335, -0.492], \
         [110.0, 0.1567, 0.000, -0.0258, 0.0000, 0.0497, -0.457], \
         [120.0, 0.0801, 0.000, -0.0311, 0.0000, 0.0740, -0.396], \
         [130.0, -0.0189, 0.000, -0.0488, 0.0101, 0.1128, -0.420], \
         [140.0, 0.0256, 0.000, -0.0422, 0.0100, 0.0889, -0.463], \
         [150.0, 0.0552, 0.000, -0.0381, 0.0109, 0.0689, -0.476], \
         [160.0, 0.0881, 0.000, -0.0306, 0.0091, 0.0366, -0.415], \
         [170.0, 0.0851, 0.000, -0.0122, 0.0025, 0.0000, -0.220], \
         [180.0, 0.0000, 0.000, 0.0000, 0.0000, 0.0000, 0.000]])


    Cx_coefficients, Cy_coefficients, Cn_coefficients = [], [], []
    for i in np.arange(1, 8):
        Cx_coefficients.append(np.interp(abs(alpha_R_degree), surge_coefficients[:, 0], surge_coefficients[:, i]))
        Cy_coefficients.append(np.interp(abs(alpha_R_degree), sway_coefficients[:, 0], sway_coefficients[:, i]))

        if i < 7:
            Cn_coefficients.append(np.interp(abs(alpha_R_degree), yaw_coefficients[:, 0], yaw_coefficients[:, i]))

    # Calculate the wind coefficients .
    C_X = -(Cx_coefficients[0] +
            Cx_coefficients[1] * ((2 * As) / Loa ** 2) +
            Cx_coefficients[2] * ((2 * Af) / B ** 2) +
            Cx_coefficients[3] * (Loa / B) +
            Cx_coefficients[4] * c / Loa +
            Cx_coefficients[5] * e / Loa +
            Cx_coefficients[6] * M)

    C_Y = -(Cy_coefficients[0] +
           Cy_coefficients[1] * 2 * As / Loa ** 2 +
           Cy_coefficients[2] * 2 * Af / B ** 2 +
           Cy_coefficients[3] * Loa / B +
           Cy_coefficients[4] * c / Loa +
           Cy_coefficients[5] * e / Loa +
           Cy_coefficients[6] * Ass / As)

    C_N = (Cn_coefficients[0] + \
          Cn_coefficients[1] * 2 * As / Loa ** 2 + \
          Cn_coefficients[2] * 2 * Af / B ** 2 + \
          Cn_coefficients[3] * Loa / B + \
          Cn_coefficients[4] * c / Loa + \
          Cn_coefficients[5] * e / Loa)

    if alpha_R_degree < 0:
        C_X = C_X
        C_Y = -C_Y
        C_N = -C_N

    X_wind = 1 / 2 * density_of_air * Af * UR ** 2 * C_X
    Y_wind = 1 / 2 * density_of_air * As * UR ** 2 * C_Y
    N_wind = 1 / 2 * density_of_air * As * Loa * UR ** 2 * C_N

    return X_wind, Y_wind, N_wind

def get_Force_Wave_X(swh, swd, swp, ship_heading, Loa):

    rho = 1025.91
    g = 9.8
    a = swh / 2
    wave_angle = (ship_heading - (swd + 180))
    wave_fre = 2 * np.pi / swp
    encounter_frequency = wave_fre  # 对于动力定位船舶，其运动速度较低，可忽略船舶自身的运动速度对波浪遭遇角频率的影响
    wave_length = 2 * np.pi * encounter_frequency ** 2 / g
    C_Xw = 0.05 - 0.2 * wave_length / Loa + 0.75 * (wave_length / Loa) ** 2 - 0.51 * (wave_length / Loa) ** 3
    X_wave = 1 / 2 * rho * g * Loa * a ** 2 * C_Xw * np.cos(np.radians(wave_angle)) * 2 * np.pi / wave_length

    return X_wave

def get_V0_plus_current_speed(ship_speed, ship_heading, current_speed, current_dir):
    ship_heading_radians = math.radians(ship_heading)
    current_dir_radians = math.radians(current_dir)
    current_dir_x = math.cos(ship_heading_radians)
    current_dir_y = math.sin(ship_heading_radians)
    V0_plus_current_speed = ship_speed + current_speed * (
                math.cos(current_dir_radians) * current_dir_x + math.sin(current_dir_radians) * current_dir_y)

    return V0_plus_current_speed

def get_ship_speed_in_ocean_env(swh, swd, swp,
                                ws, wd,
                                cs, cd,
                                ship_speed_ms, ship_heading, Loa):
    Rt = get_Rt(ship_speed_ms)
    T = Rt

    X_wave = get_Force_Wave_X(swh, swd, swp, ship_heading, Loa)
    X_wind, Y_wind, N_wind = Force_and_moment_of_wind(ws, wd, ship_speed_ms, ship_heading)
    youxiao_T = Rt - X_wind - X_wave
    V = get_V(youxiao_T)

    V0_plus_cs_ms = get_V0_plus_current_speed(V, ship_heading, cs, cd)

    return V0_plus_cs_ms