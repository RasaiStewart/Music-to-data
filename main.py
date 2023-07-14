# Necessary libraries to be imported
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import wavfile
import mpmath

# Musical notes
# The first step, after the imports, consists of creating a musical scale or notes.
# This is required to create a pleasant melody. Without it, the sound will feel like noise.


# -- Create the list of musical notes.
scale = []
for k in range(35, 65):
    note=440*2**((k-49)/12)
    if k%12 != 0 and k%12 !=2 and k%12 !=5 and k%12 !=7 and k%12 != 10:
        # add musical note (skip the half tones).
        scale.append(note)
# number of musical notes.
n_notes = len(scale)


# Data production and transformation
# The second step generated the data and transforms it via rescaling, so that it can
# easily be turned into music. Heere sample values of the Dirichlet eta function
# (a sister of the Riemann Zeta function) as input date. The data is then transformed
# into multivariate data using 3 features indexed by time: frequency
# (the pitch), volume (amplitude), and the duration for each of the 300 musical notes
# corresponding to the data. Real and Imag are respectively the real
# and imaginary part of a complex number.

# -- Generate the data

n = 300
sigma = 0.5
min_t = 400000
max_t = 400020


def create_data(f, nobs, min_t, max_t, sigma):
    z_real = []
    z_imag = []
    z_modulus = []
    incr_t = (max_t - min_t) / nobs
    for t in np.arange(min_t, max_t, incr_t):
        if f == 'Zeta':
            z = mpmath.zeta(complex(sigma, t))
        elif f == 'Eta':
            z = mpmath.altzeta(complex(sigma, t))
        z_real.append(float(z.real))
        z_imag.append(float(z.imag))
        modulus = np.sqrt(z.real * z.real + z.imag * z.imag)
        z_modulus.append(float(modulus))
    return (z_real, z_imag, z_modulus)


(z_real, z_imag, z_modulus) = create_data('Eta', n, min_t, max_t, sigma)

# should be identical to nobs
size = len(z_real)
x = np.arange(size)

# frequency of each note
y = z_real
min = np.min(y)
max = np.max(y)
yf = 0.999 * n_notes * (y - min) / (max - min)

# duration of each note
z = z_imag
min = np.min(z)
max = np.max(z)
zf = 0.1 + 0.4 * (z - min) / (max - min)

# volume of each note
v = z_modulus
min = np.min(v)
max = np.max(v)
vf = 500 + 2000 * (1 - (v - min) / (max - min))


# Plotting the sound waves
# The next step is to plot the 3 values attached
# to each musical note, as 3 time series

#-- plot data

mpl.rcParams['axes.linewidth'] = 0.3
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=7)
plt.rcParams['axes.linewidth'] = 0.1
plt.plot(x, y, color='red', linewidth = 0.3)
plt.plot(x, z, color='blue', linewidth = 0.3)
plt.plot(x, v, color='green', linewidth = 0.3)
plt.legend(['frequency','duration','volume'], fontsize="7",
    loc ="upper center", ncol=3)
plt.show()


# Producing the sound track
# Each wave corresponds to a musical note. The concatenated waves are turned into
# a wav file using the wavfile.write function from the scipy library.

#-- Turn the data into music

def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate*duration))
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

wave=[]
for t in x: # loop over dataset observations, create one note per observation
    note = int(yf[t])
    duration = zf[t]
    frequency = scale[note]
    volume = vf[t]  ## 2048
    new_wave = get_sine_wave(frequency, duration = zf[t], amplitude = vf[t])
    wave = np.concatenate((wave,new_wave))
wavfile.write('sound.wav', rate=44100, data=wave.astype(np.int16))