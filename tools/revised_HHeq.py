import matplotlib.pyplot as plt
import numpy as np

# import japanize_matplotlib

E_REST = -65.0
C = 1.0
G_LEAK = 0.3
E_LEAK = 10.6 + E_REST
G_NA = 120.0
E_NA = 115.0 + E_REST
G_K = 36.0
E_K = -12.0 + E_REST
i_ext = 9

DT = 0.01
T = 100
NT = int(T / DT)

formula = {"alpha": {"m": (lambda v: (2.5 - 0.1 * (v - E_REST)) / (np.exp(2.5 - 0.1 * (v - E_REST)) - 1)),
                     "h": (lambda v: 0.07 * np.exp(-(v - E_REST) / 20)),
                     "n": (lambda v: (0.1 - 0.01 * (v - E_REST)) / (np.exp(1 - 0.1 * (v - E_REST)) - 1)),
                     },
           "beta": {"m": (lambda v: 4 * np.exp(-(v - E_REST) / 18)),
                    "h": (lambda v: 1 / (np.exp(3 - 0.1 * (v - E_REST)) + 1)),
                    "n": (lambda v: 0.125 * np.exp(- (v - E_REST) / 80))
                    }
           }


samplesv = ["m","h","n","v"]
samplesvt = ["m","h","n","v","t"]



def zero(key, v):
    return formula["alpha"][key](v) / (formula["alpha"][key](v) + formula["beta"][key](v))

def tau(key,v):
    return 1.0/(formula["alpha"][key](v) + formula["beta"][key](v))

def diff_mhn(key,v,m):
    return (1.0/tau(key,v)) * (-m + zero(key,v))

def diff_v(v, m, h, n, i_ext):
    return (-G_LEAK * (v - E_LEAK) - G_NA * (m ** 3) * h * (v - E_NA) - G_K * (n ** 4) * (v - E_K) + i_ext) / C

def diff1(key,i,rec):
    if key == "v":
        return diff_v(rec["v"][i], rec["m"][i], rec["h"][i], rec["n"][i], i_ext)
    else:
        return diff_mhn(key, rec["v"][i], rec[key][i])

def diff2(key,i,rec):
    if key == "v":
        return diff_v(rec["v"][i] + 0.5 * DT * diff1("v",i,rec), rec["m"][i] + 0.5 * DT * diff1("m",i,rec), rec["h"][i] + 0.5 * DT * diff1("h",i,rec), rec["n"][i] + 0.5 * DT * diff1("n",i,rec), i_ext)
    else:
        return diff_mhn(key, rec["v"][i] + 0.5 * DT * diff1("v",i,rec), rec[key][i] + 0.5 * DT * diff1(key,i,rec))

def diff3(key,i,rec):
    if key == "v":
        return diff_v(rec["v"][i] + 0.5 * DT * diff2("v",i,rec), rec["m"][i] + 0.5 * DT * diff2("m",i,rec), rec["h"][i] + 0.5 * DT * diff2("h",i,rec), rec["n"][i] + 0.5 * DT * diff2("n",i,rec), i_ext)
    else:
        return diff_mhn(key, rec["v"][i] + 0.5 * DT * diff2("v",i,rec), rec[key][i] + 0.5 * DT * diff2(key,i,rec))

def diff4(key,i,rec):
    if key == "v":
        return diff_v(rec["v"][i] + DT * diff2("v",i,rec), rec["m"][i] + DT * diff2("m",i,rec), rec["h"][i] + DT * diff2("h",i,rec), rec["n"][i] + DT * diff2("n",i,rec), i_ext)
    else:
        return diff_mhn(key, rec["v"][i] + DT * diff2("v",i,rec), rec[key][i] + DT * diff2(key,i,rec))

def all_diff(key,i,rec):
    return DT * (diff1(key,i,rec) + 2 * diff2(key,i,rec) + 2 * diff3(key,i,rec) + diff4(key,i,rec)) / 6


def initial(key):
    if key == "v":
        return E_REST
    else:
        return zero(key,E_REST)

def display(rec,x,y):
    plt.plot(rec[x],rec[y])
    plt.show()

def main_rev(x="t",y="v"):
    rec = {key: np.zeros(NT + 1) for key in samplesvt}
    for sample in samplesv:
        rec[sample][0] = initial(sample)
    for i in range(NT):
        t = DT * i
        rec["t"][i+1] = t
        for sample in samplesv:
            rec[sample][i+1] = rec[sample][i] + all_diff(sample,i,rec)
    display(rec,x,y)
    return rec

if __name__ == "__main__":
    main_rev()







