
====================================================================
       AD-RaNN feature-number x random-seed experiment
====================================================================
Training feature numbers = 200 400 600 800 1000 
Expanded final features = 1000
Seeds                   = 1 : 100
Total training runs     = 500
Initial p               = [4.000000, 8.000000]
lambda                  = 1.000e-07
Fast evaluator          = 1

For each run:
  1. Optimize p using m features.
  2. Final unregularized LS using the same m features.
  3. Fix p* and perform another unregularized LS using 1000 features.

IMPORTANT:
  Original total time does NOT include the m -> 1000 refit.
====================================================================


####################################################################
Feature number 1 / 5
Training m = 200
Expanded final m = 1000
Initial p = [4.000000, 8.000000]
lambda = 1.000e-07
####################################################################

[m= 200 | seed   1/100] ... p*=[1.018169, 4.661507] | L2(m)=9.150e-04 | Linf(m)=3.163e-03 | checkpoint=4 | opt=1.305 s | original total=1.352 s | L2(1000)=2.074e-09 | Linf(1000)=1.225e-08 | extra 1000-refit=0.279 s
[m= 200 | seed   2/100] ... p*=[1.753860, 4.949426] | L2(m)=2.308e-03 | Linf(m)=9.659e-03 | checkpoint=50 | opt=1.217 s | original total=1.250 s | L2(1000)=1.638e-09 | Linf(1000)=6.304e-09 | extra 1000-refit=0.290 s
[m= 200 | seed   3/100] ... p*=[1.178222, 6.546027] | L2(m)=2.820e-03 | Linf(m)=7.016e-03 | checkpoint=10 | opt=1.188 s | original total=1.224 s | L2(1000)=1.580e-09 | Linf(1000)=8.041e-09 | extra 1000-refit=0.282 s
[m= 200 | seed   4/100] ... p*=[0.865449, 5.057355] | L2(m)=7.281e-04 | Linf(m)=3.014e-03 | checkpoint=50 | opt=1.246 s | original total=1.281 s | L2(1000)=1.578e-09 | Linf(1000)=7.399e-09 | extra 1000-refit=0.252 s
[m= 200 | seed   5/100] ... p*=[1.069994, 4.794461] | L2(m)=1.194e-03 | Linf(m)=4.234e-03 | checkpoint=4 | opt=1.148 s | original total=1.191 s | L2(1000)=1.389e-09 | Linf(1000)=9.497e-09 | extra 1000-refit=0.272 s
[m= 200 | seed   6/100] ... p*=[1.114026, 6.960052] | L2(m)=3.292e-03 | Linf(m)=1.230e-02 | checkpoint=50 | opt=1.158 s | original total=1.187 s | L2(1000)=6.001e-10 | Linf(1000)=2.342e-09 | extra 1000-refit=0.255 s
[m= 200 | seed   7/100] ... p*=[1.175439, 5.617240] | L2(m)=2.187e-03 | Linf(m)=7.061e-03 | checkpoint=3 | opt=1.193 s | original total=1.224 s | L2(1000)=1.044e-09 | Linf(1000)=5.981e-09 | extra 1000-refit=0.265 s
[m= 200 | seed   8/100] ... p*=[0.855332, 7.502871] | L2(m)=6.958e-03 | Linf(m)=1.460e-02 | checkpoint=9 | opt=1.124 s | original total=1.158 s | L2(1000)=3.220e-09 | Linf(1000)=1.057e-08 | extra 1000-refit=0.269 s
[m= 200 | seed   9/100] ... p*=[0.776080, 5.120066] | L2(m)=1.082e-03 | Linf(m)=3.313e-03 | checkpoint=4 | opt=1.137 s | original total=1.167 s | L2(1000)=2.042e-09 | Linf(1000)=1.120e-08 | extra 1000-refit=0.250 s
[m= 200 | seed  10/100] ... p*=[1.197333, 4.203788] | L2(m)=1.336e-03 | Linf(m)=5.055e-03 | checkpoint=5 | opt=1.146 s | original total=1.177 s | L2(1000)=8.686e-10 | Linf(1000)=5.205e-09 | extra 1000-refit=0.283 s
[m= 200 | seed  11/100] ... p*=[0.892871, 7.106441] | L2(m)=4.456e-03 | Linf(m)=1.051e-02 | checkpoint=50 | opt=1.090 s | original total=1.118 s | L2(1000)=3.881e-09 | Linf(1000)=7.857e-09 | extra 1000-refit=0.260 s
[m= 200 | seed  12/100] ... p*=[0.993326, 4.529184] | L2(m)=1.759e-03 | Linf(m)=3.984e-03 | checkpoint=4 | opt=1.119 s | original total=1.154 s | L2(1000)=1.640e-09 | Linf(1000)=6.349e-09 | extra 1000-refit=0.269 s
[m= 200 | seed  13/100] ... p*=[1.633452, 7.945116] | L2(m)=8.566e-03 | Linf(m)=2.936e-02 | checkpoint=50 | opt=1.075 s | original total=1.102 s | L2(1000)=1.533e-09 | Linf(1000)=5.028e-09 | extra 1000-refit=0.251 s
[m= 200 | seed  14/100] ... p*=[1.414236, 3.928356] | L2(m)=2.645e-03 | Linf(m)=8.521e-03 | checkpoint=44 | opt=1.146 s | original total=1.175 s | L2(1000)=4.120e-09 | Linf(1000)=2.298e-08 | extra 1000-refit=0.303 s
[m= 200 | seed  15/100] ... p*=[0.840526, 4.554502] | L2(m)=5.269e-04 | Linf(m)=1.940e-03 | checkpoint=41 | opt=1.151 s | original total=1.181 s | L2(1000)=1.307e-09 | Linf(1000)=7.763e-09 | extra 1000-refit=0.269 s
[m= 200 | seed  16/100] ... p*=[1.025660, 4.986573] | L2(m)=1.490e-03 | Linf(m)=3.224e-03 | checkpoint=4 | opt=1.129 s | original total=1.157 s | L2(1000)=1.153e-09 | Linf(1000)=4.915e-09 | extra 1000-refit=0.237 s
[m= 200 | seed  17/100] ... p*=[1.268759, 6.335114] | L2(m)=3.167e-03 | Linf(m)=1.287e-02 | checkpoint=44 | opt=1.076 s | original total=1.104 s | L2(1000)=1.025e-09 | Linf(1000)=3.893e-09 | extra 1000-refit=0.260 s
[m= 200 | seed  18/100] ... p*=[1.155413, 5.178616] | L2(m)=1.690e-03 | Linf(m)=5.889e-03 | checkpoint=4 | opt=1.221 s | original total=1.250 s | L2(1000)=7.817e-10 | Linf(1000)=3.432e-09 | extra 1000-refit=0.266 s
[m= 200 | seed  19/100] ... p*=[1.153608, 4.736931] | L2(m)=8.233e-04 | Linf(m)=3.527e-03 | checkpoint=4 | opt=1.104 s | original total=1.132 s | L2(1000)=1.753e-09 | Linf(1000)=7.123e-09 | extra 1000-refit=0.274 s
[m= 200 | seed  20/100] ... p*=[1.695139, 8.999541] | L2(m)=9.793e-03 | Linf(m)=1.772e-02 | checkpoint=11 | opt=1.117 s | original total=1.143 s | L2(1000)=1.416e-09 | Linf(1000)=4.215e-09 | extra 1000-refit=0.259 s
[m= 200 | seed  21/100] ... p*=[0.901326, 8.200673] | L2(m)=9.808e-03 | Linf(m)=1.932e-02 | checkpoint=36 | opt=1.144 s | original total=1.174 s | L2(1000)=7.765e-09 | Linf(1000)=1.986e-08 | extra 1000-refit=0.227 s
[m= 200 | seed  22/100] ... p*=[1.251946, 6.250696] | L2(m)=1.773e-03 | Linf(m)=7.133e-03 | checkpoint=49 | opt=1.110 s | original total=1.140 s | L2(1000)=6.919e-10 | Linf(1000)=4.339e-09 | extra 1000-refit=0.239 s
[m= 200 | seed  23/100] ... p*=[0.931928, 4.778553] | L2(m)=7.180e-04 | Linf(m)=2.489e-03 | checkpoint=4 | opt=1.165 s | original total=1.195 s | L2(1000)=1.938e-09 | Linf(1000)=1.036e-08 | extra 1000-refit=0.258 s
[m= 200 | seed  24/100] ... p*=[1.119202, 3.372225] | L2(m)=8.027e-04 | Linf(m)=3.911e-03 | checkpoint=39 | opt=1.169 s | original total=1.199 s | L2(1000)=6.626e-09 | Linf(1000)=3.769e-08 | extra 1000-refit=0.260 s
[m= 200 | seed  25/100] ... p*=[0.903568, 5.105610] | L2(m)=8.770e-04 | Linf(m)=2.771e-03 | checkpoint=4 | opt=1.177 s | original total=1.209 s | L2(1000)=1.550e-09 | Linf(1000)=9.665e-09 | extra 1000-refit=0.274 s
[m= 200 | seed  26/100] ... p*=[1.409691, 4.805287] | L2(m)=2.483e-03 | Linf(m)=9.607e-03 | checkpoint=35 | opt=1.175 s | original total=1.204 s | L2(1000)=1.018e-09 | Linf(1000)=5.488e-09 | extra 1000-refit=0.247 s
[m= 200 | seed  27/100] ... p*=[1.039469, 4.783816] | L2(m)=1.369e-03 | Linf(m)=5.652e-03 | checkpoint=4 | opt=1.118 s | original total=1.148 s | L2(1000)=1.405e-09 | Linf(1000)=9.176e-09 | extra 1000-refit=0.272 s
[m= 200 | seed  28/100] ... p*=[0.962401, 5.061816] | L2(m)=6.709e-04 | Linf(m)=1.460e-03 | checkpoint=4 | opt=1.152 s | original total=1.181 s | L2(1000)=1.155e-09 | Linf(1000)=5.434e-09 | extra 1000-refit=0.253 s
[m= 200 | seed  29/100] ... p*=[1.021118, 5.699875] | L2(m)=1.484e-03 | Linf(m)=4.538e-03 | checkpoint=46 | opt=1.194 s | original total=1.224 s | L2(1000)=1.082e-09 | Linf(1000)=7.306e-09 | extra 1000-refit=0.258 s
[m= 200 | seed  30/100] ... p*=[0.793623, 7.816342] | L2(m)=4.209e-03 | Linf(m)=1.044e-02 | checkpoint=50 | opt=1.149 s | original total=1.177 s | L2(1000)=4.972e-09 | Linf(1000)=1.427e-08 | extra 1000-refit=0.246 s
[m= 200 | seed  31/100] ... p*=[0.984972, 6.724618] | L2(m)=1.743e-03 | Linf(m)=4.153e-03 | checkpoint=49 | opt=1.118 s | original total=1.146 s | L2(1000)=1.086e-09 | Linf(1000)=3.221e-09 | extra 1000-refit=0.238 s
[m= 200 | seed  32/100] ... p*=[1.049980, 4.642829] | L2(m)=9.416e-04 | Linf(m)=2.814e-03 | checkpoint=46 | opt=1.202 s | original total=1.236 s | L2(1000)=1.456e-09 | Linf(1000)=1.009e-08 | extra 1000-refit=0.266 s
[m= 200 | seed  33/100] ... p*=[0.868913, 3.991042] | L2(m)=2.962e-04 | Linf(m)=7.570e-04 | checkpoint=7 | opt=1.183 s | original total=1.214 s | L2(1000)=2.940e-09 | Linf(1000)=1.617e-08 | extra 1000-refit=0.268 s
[m= 200 | seed  34/100] ... p*=[0.968877, 3.605797] | L2(m)=4.490e-04 | Linf(m)=1.523e-03 | checkpoint=8 | opt=1.162 s | original total=1.194 s | L2(1000)=4.082e-09 | Linf(1000)=2.512e-08 | extra 1000-refit=0.256 s
[m= 200 | seed  35/100] ... p*=[1.097441, 4.848401] | L2(m)=7.848e-04 | Linf(m)=2.558e-03 | checkpoint=50 | opt=1.144 s | original total=1.176 s | L2(1000)=5.049e-10 | Linf(1000)=2.213e-09 | extra 1000-refit=0.261 s
[m= 200 | seed  36/100] ... p*=[0.980141, 6.385896] | L2(m)=2.426e-03 | Linf(m)=5.937e-03 | checkpoint=4 | opt=1.158 s | original total=1.191 s | L2(1000)=1.403e-09 | Linf(1000)=6.064e-09 | extra 1000-refit=0.263 s
[m= 200 | seed  37/100] ... p*=[1.432013, 9.167252] | L2(m)=1.038e-02 | Linf(m)=4.010e-02 | checkpoint=41 | opt=1.112 s | original total=1.139 s | L2(1000)=1.570e-09 | Linf(1000)=4.631e-09 | extra 1000-refit=0.249 s
[m= 200 | seed  38/100] ... p*=[0.992723, 4.710149] | L2(m)=1.196e-03 | Linf(m)=2.855e-03 | checkpoint=4 | opt=1.141 s | original total=1.173 s | L2(1000)=2.344e-09 | Linf(1000)=1.032e-08 | extra 1000-refit=0.261 s
[m= 200 | seed  39/100] ... p*=[1.089165, 5.953011] | L2(m)=1.587e-03 | Linf(m)=5.189e-03 | checkpoint=8 | opt=1.132 s | original total=1.162 s | L2(1000)=8.163e-10 | Linf(1000)=5.825e-09 | extra 1000-refit=0.264 s
[m= 200 | seed  40/100] ... p*=[0.773175, 4.928878] | L2(m)=1.461e-03 | Linf(m)=3.470e-03 | checkpoint=4 | opt=1.178 s | original total=1.210 s | L2(1000)=2.163e-09 | Linf(1000)=8.026e-09 | extra 1000-refit=0.278 s
[m= 200 | seed  41/100] ... p*=[0.983199, 4.506393] | L2(m)=7.008e-04 | Linf(m)=2.915e-03 | checkpoint=5 | opt=1.207 s | original total=1.236 s | L2(1000)=1.522e-09 | Linf(1000)=6.510e-09 | extra 1000-refit=0.283 s
[m= 200 | seed  42/100] ... p*=[0.817184, 4.299399] | L2(m)=9.516e-04 | Linf(m)=2.732e-03 | checkpoint=39 | opt=1.228 s | original total=1.258 s | L2(1000)=2.491e-09 | Linf(1000)=9.465e-09 | extra 1000-refit=0.282 s
[m= 200 | seed  43/100] ... p*=[1.133689, 6.274683] | L2(m)=1.786e-03 | Linf(m)=5.828e-03 | checkpoint=33 | opt=1.123 s | original total=1.149 s | L2(1000)=7.728e-10 | Linf(1000)=3.214e-09 | extra 1000-refit=0.249 s
[m= 200 | seed  44/100] ... p*=[0.977747, 5.010784] | L2(m)=6.571e-04 | Linf(m)=2.199e-03 | checkpoint=4 | opt=1.126 s | original total=1.154 s | L2(1000)=1.225e-09 | Linf(1000)=5.676e-09 | extra 1000-refit=0.247 s
[m= 200 | seed  45/100] ... p*=[1.021932, 4.915952] | L2(m)=2.441e-03 | Linf(m)=7.055e-03 | checkpoint=9 | opt=1.157 s | original total=1.186 s | L2(1000)=1.020e-09 | Linf(1000)=5.613e-09 | extra 1000-refit=0.258 s
[m= 200 | seed  46/100] ... p*=[1.306297, 4.621758] | L2(m)=1.288e-03 | Linf(m)=4.736e-03 | checkpoint=4 | opt=1.093 s | original total=1.122 s | L2(1000)=1.290e-09 | Linf(1000)=5.890e-09 | extra 1000-refit=0.261 s
[m= 200 | seed  47/100] ... p*=[1.093885, 5.753771] | L2(m)=2.582e-03 | Linf(m)=1.014e-02 | checkpoint=44 | opt=1.159 s | original total=1.187 s | L2(1000)=1.660e-09 | Linf(1000)=7.237e-09 | extra 1000-refit=0.272 s
[m= 200 | seed  48/100] ... p*=[1.396499, 11.530687] | L2(m)=1.956e-02 | Linf(m)=4.495e-02 | checkpoint=41 | opt=1.138 s | original total=1.168 s | L2(1000)=1.193e-08 | Linf(1000)=6.270e-08 | extra 1000-refit=0.239 s
[m= 200 | seed  49/100] ... p*=[1.258587, 6.106455] | L2(m)=1.383e-03 | Linf(m)=3.969e-03 | checkpoint=50 | opt=1.113 s | original total=1.141 s | L2(1000)=8.212e-10 | Linf(1000)=4.442e-09 | extra 1000-refit=0.249 s
[m= 200 | seed  50/100] ... p*=[1.315815, 4.855910] | L2(m)=1.328e-03 | Linf(m)=3.653e-03 | checkpoint=50 | opt=1.148 s | original total=1.182 s | L2(1000)=1.218e-09 | Linf(1000)=5.836e-09 | extra 1000-refit=0.269 s
[m= 200 | seed  51/100] ... p*=[1.317961, 8.693026] | L2(m)=9.384e-03 | Linf(m)=2.226e-02 | checkpoint=47 | opt=1.142 s | original total=1.171 s | L2(1000)=2.104e-09 | Linf(1000)=1.018e-08 | extra 1000-refit=0.263 s
[m= 200 | seed  52/100] ... p*=[1.231180, 5.535156] | L2(m)=4.403e-03 | Linf(m)=2.002e-02 | checkpoint=50 | opt=1.116 s | original total=1.145 s | L2(1000)=1.299e-09 | Linf(1000)=5.997e-09 | extra 1000-refit=0.259 s
[m= 200 | seed  53/100] ... p*=[0.841464, 5.934827] | L2(m)=1.891e-03 | Linf(m)=4.321e-03 | checkpoint=7 | opt=1.150 s | original total=1.182 s | L2(1000)=1.588e-09 | Linf(1000)=6.510e-09 | extra 1000-refit=0.247 s
[m= 200 | seed  54/100] ... p*=[1.490986, 9.601704] | L2(m)=6.496e-03 | Linf(m)=1.540e-02 | checkpoint=45 | opt=1.113 s | original total=1.142 s | L2(1000)=1.963e-09 | Linf(1000)=3.303e-09 | extra 1000-refit=0.255 s
[m= 200 | seed  55/100] ... p*=[0.960571, 5.131043] | L2(m)=1.265e-03 | Linf(m)=4.542e-03 | checkpoint=4 | opt=1.271 s | original total=1.302 s | L2(1000)=1.031e-09 | Linf(1000)=5.644e-09 | extra 1000-refit=0.261 s
[m= 200 | seed  56/100] ... p*=[0.736133, 5.819211] | L2(m)=1.580e-03 | Linf(m)=2.464e-03 | checkpoint=43 | opt=1.222 s | original total=1.253 s | L2(1000)=3.203e-09 | Linf(1000)=9.595e-09 | extra 1000-refit=0.248 s
[m= 200 | seed  57/100] ... p*=[0.762793, 4.832121] | L2(m)=9.823e-04 | Linf(m)=2.776e-03 | checkpoint=7 | opt=1.133 s | original total=1.163 s | L2(1000)=2.347e-09 | Linf(1000)=9.149e-09 | extra 1000-refit=0.235 s
[m= 200 | seed  58/100] ... p*=[0.939284, 4.186496] | L2(m)=5.259e-04 | Linf(m)=2.382e-03 | checkpoint=46 | opt=1.165 s | original total=1.198 s | L2(1000)=3.801e-09 | Linf(1000)=2.639e-08 | extra 1000-refit=0.253 s
[m= 200 | seed  59/100] ... p*=[0.848577, 5.787898] | L2(m)=1.026e-03 | Linf(m)=2.374e-03 | checkpoint=42 | opt=1.177 s | original total=1.209 s | L2(1000)=2.341e-09 | Linf(1000)=1.468e-08 | extra 1000-refit=0.288 s
[m= 200 | seed  60/100] ... p*=[0.978603, 4.983380] | L2(m)=1.239e-03 | Linf(m)=3.902e-03 | checkpoint=7 | opt=1.134 s | original total=1.168 s | L2(1000)=2.339e-09 | Linf(1000)=1.019e-08 | extra 1000-refit=0.277 s
[m= 200 | seed  61/100] ... p*=[1.010077, 4.765298] | L2(m)=5.031e-04 | Linf(m)=1.269e-03 | checkpoint=7 | opt=1.161 s | original total=1.193 s | L2(1000)=1.185e-09 | Linf(1000)=5.438e-09 | extra 1000-refit=0.229 s
[m= 200 | seed  62/100] ... p*=[1.296535, 5.001726] | L2(m)=2.823e-03 | Linf(m)=1.280e-02 | checkpoint=44 | opt=1.186 s | original total=1.214 s | L2(1000)=1.107e-09 | Linf(1000)=5.093e-09 | extra 1000-refit=0.264 s
[m= 200 | seed  63/100] ... p*=[0.942103, 4.971058] | L2(m)=1.165e-03 | Linf(m)=3.678e-03 | checkpoint=8 | opt=1.129 s | original total=1.161 s | L2(1000)=6.789e-10 | Linf(1000)=2.950e-09 | extra 1000-refit=0.247 s
[m= 200 | seed  64/100] ... p*=[0.927297, 6.024573] | L2(m)=1.922e-03 | Linf(m)=2.865e-03 | checkpoint=50 | opt=1.113 s | original total=1.142 s | L2(1000)=1.892e-09 | Linf(1000)=1.253e-08 | extra 1000-refit=0.262 s
[m= 200 | seed  65/100] ... p*=[1.437568, 5.176175] | L2(m)=1.010e-03 | Linf(m)=2.821e-03 | checkpoint=21 | opt=1.160 s | original total=1.189 s | L2(1000)=6.682e-10 | Linf(1000)=2.557e-09 | extra 1000-refit=0.238 s
[m= 200 | seed  66/100] ... p*=[0.827814, 3.684200] | L2(m)=3.664e-04 | Linf(m)=1.048e-03 | checkpoint=5 | opt=1.197 s | original total=1.232 s | L2(1000)=4.748e-09 | Linf(1000)=3.363e-08 | extra 1000-refit=0.262 s
[m= 200 | seed  67/100] ... p*=[1.269028, 5.919264] | L2(m)=2.569e-03 | Linf(m)=9.191e-03 | checkpoint=50 | opt=1.106 s | original total=1.134 s | L2(1000)=1.437e-09 | Linf(1000)=7.727e-09 | extra 1000-refit=0.258 s
[m= 200 | seed  68/100] ... p*=[0.874490, 4.377547] | L2(m)=1.823e-04 | Linf(m)=6.573e-04 | checkpoint=40 | opt=1.254 s | original total=1.283 s | L2(1000)=2.802e-09 | Linf(1000)=1.070e-08 | extra 1000-refit=0.285 s
[m= 200 | seed  69/100] ... p*=[0.863643, 4.681377] | L2(m)=7.908e-04 | Linf(m)=3.012e-03 | checkpoint=50 | opt=1.184 s | original total=1.215 s | L2(1000)=2.186e-09 | Linf(1000)=9.056e-09 | extra 1000-refit=0.258 s
[m= 200 | seed  70/100] ... p*=[1.689159, 7.468149] | L2(m)=1.049e-02 | Linf(m)=3.344e-02 | checkpoint=50 | opt=1.124 s | original total=1.152 s | L2(1000)=6.577e-10 | Linf(1000)=3.228e-09 | extra 1000-refit=0.282 s
[m= 200 | seed  71/100] ... p*=[1.828810, 9.253244] | L2(m)=7.105e-03 | Linf(m)=2.134e-02 | checkpoint=30 | opt=1.115 s | original total=1.145 s | L2(1000)=9.474e-10 | Linf(1000)=2.673e-09 | extra 1000-refit=0.244 s
[m= 200 | seed  72/100] ... p*=[0.987551, 6.277643] | L2(m)=1.174e-03 | Linf(m)=3.127e-03 | checkpoint=44 | opt=1.128 s | original total=1.159 s | L2(1000)=9.701e-10 | Linf(1000)=4.482e-09 | extra 1000-refit=0.263 s
[m= 200 | seed  73/100] ... p*=[0.772671, 4.169500] | L2(m)=5.824e-04 | Linf(m)=1.511e-03 | checkpoint=14 | opt=1.223 s | original total=1.253 s | L2(1000)=3.157e-09 | Linf(1000)=1.911e-08 | extra 1000-refit=0.232 s
[m= 200 | seed  74/100] ... p*=[0.954061, 6.321484] | L2(m)=3.248e-03 | Linf(m)=9.328e-03 | checkpoint=42 | opt=1.197 s | original total=1.226 s | L2(1000)=2.224e-09 | Linf(1000)=8.607e-09 | extra 1000-refit=0.263 s
[m= 200 | seed  75/100] ... p*=[1.046270, 6.109256] | L2(m)=1.920e-03 | Linf(m)=7.566e-03 | checkpoint=44 | opt=1.132 s | original total=1.162 s | L2(1000)=8.436e-10 | Linf(1000)=5.061e-09 | extra 1000-refit=0.261 s
[m= 200 | seed  76/100] ... p*=[1.242536, 4.497153] | L2(m)=1.136e-03 | Linf(m)=3.831e-03 | checkpoint=50 | opt=1.194 s | original total=1.224 s | L2(1000)=1.394e-09 | Linf(1000)=6.356e-09 | extra 1000-refit=0.286 s
[m= 200 | seed  77/100] ... p*=[1.249988, 4.482820] | L2(m)=1.923e-03 | Linf(m)=9.825e-03 | checkpoint=50 | opt=1.146 s | original total=1.178 s | L2(1000)=1.665e-09 | Linf(1000)=7.530e-09 | extra 1000-refit=0.257 s
[m= 200 | seed  78/100] ... p*=[1.490790, 5.324181] | L2(m)=1.383e-03 | Linf(m)=5.998e-03 | checkpoint=46 | opt=1.175 s | original total=1.205 s | L2(1000)=1.061e-09 | Linf(1000)=4.455e-09 | extra 1000-refit=0.291 s
[m= 200 | seed  79/100] ... p*=[1.058873, 4.950391] | L2(m)=7.923e-04 | Linf(m)=2.638e-03 | checkpoint=4 | opt=1.144 s | original total=1.177 s | L2(1000)=1.251e-09 | Linf(1000)=8.235e-09 | extra 1000-refit=0.251 s
[m= 200 | seed  80/100] ... p*=[1.146615, 6.618020] | L2(m)=2.212e-03 | Linf(m)=6.454e-03 | checkpoint=41 | opt=1.106 s | original total=1.137 s | L2(1000)=7.114e-10 | Linf(1000)=2.831e-09 | extra 1000-refit=0.278 s
[m= 200 | seed  81/100] ... p*=[0.993128, 8.099456] | L2(m)=2.472e-03 | Linf(m)=7.369e-03 | checkpoint=50 | opt=1.125 s | original total=1.151 s | L2(1000)=2.583e-09 | Linf(1000)=6.876e-09 | extra 1000-refit=0.253 s
[m= 200 | seed  82/100] ... p*=[1.039056, 7.322587] | L2(m)=4.647e-03 | Linf(m)=1.518e-02 | checkpoint=10 | opt=1.139 s | original total=1.167 s | L2(1000)=1.229e-09 | Linf(1000)=4.603e-09 | extra 1000-refit=0.263 s
[m= 200 | seed  83/100] ... p*=[1.291979, 5.959889] | L2(m)=1.682e-03 | Linf(m)=6.958e-03 | checkpoint=50 | opt=1.097 s | original total=1.128 s | L2(1000)=6.726e-10 | Linf(1000)=4.198e-09 | extra 1000-refit=0.251 s
[m= 200 | seed  84/100] ... p*=[0.919880, 7.066583] | L2(m)=2.979e-03 | Linf(m)=9.813e-03 | checkpoint=48 | opt=1.120 s | original total=1.149 s | L2(1000)=2.351e-09 | Linf(1000)=6.646e-09 | extra 1000-refit=0.237 s
[m= 200 | seed  85/100] ... p*=[1.264476, 7.120311] | L2(m)=4.239e-03 | Linf(m)=9.366e-03 | checkpoint=36 | opt=1.133 s | original total=1.160 s | L2(1000)=7.993e-10 | Linf(1000)=2.621e-09 | extra 1000-refit=0.240 s
[m= 200 | seed  86/100] ... p*=[1.631446, 6.091543] | L2(m)=6.102e-03 | Linf(m)=2.331e-02 | checkpoint=50 | opt=1.121 s | original total=1.149 s | L2(1000)=6.407e-10 | Linf(1000)=4.038e-09 | extra 1000-refit=0.276 s
[m= 200 | seed  87/100] ... p*=[0.959770, 4.887105] | L2(m)=7.068e-04 | Linf(m)=1.873e-03 | checkpoint=4 | opt=1.172 s | original total=1.200 s | L2(1000)=2.756e-09 | Linf(1000)=1.600e-08 | extra 1000-refit=0.247 s
[m= 200 | seed  88/100] ... p*=[0.891004, 5.461893] | L2(m)=1.281e-03 | Linf(m)=3.047e-03 | checkpoint=49 | opt=1.151 s | original total=1.179 s | L2(1000)=1.688e-09 | Linf(1000)=6.348e-09 | extra 1000-refit=0.262 s
[m= 200 | seed  89/100] ... p*=[1.056592, 5.597488] | L2(m)=1.569e-03 | Linf(m)=3.887e-03 | checkpoint=50 | opt=1.079 s | original total=1.107 s | L2(1000)=9.902e-10 | Linf(1000)=3.875e-09 | extra 1000-refit=0.289 s
[m= 200 | seed  90/100] ... p*=[1.107604, 6.179098] | L2(m)=2.179e-03 | Linf(m)=7.633e-03 | checkpoint=42 | opt=1.178 s | original total=1.208 s | L2(1000)=5.132e-10 | Linf(1000)=2.401e-09 | extra 1000-refit=0.277 s
[m= 200 | seed  91/100] ... p*=[1.003834, 4.958805] | L2(m)=8.031e-04 | Linf(m)=3.199e-03 | checkpoint=7 | opt=1.123 s | original total=1.153 s | L2(1000)=1.472e-09 | Linf(1000)=9.512e-09 | extra 1000-refit=0.281 s
[m= 200 | seed  92/100] ... p*=[1.075637, 4.044818] | L2(m)=1.328e-03 | Linf(m)=6.206e-03 | checkpoint=8 | opt=1.244 s | original total=1.275 s | L2(1000)=2.338e-09 | Linf(1000)=1.197e-08 | extra 1000-refit=0.255 s
[m= 200 | seed  93/100] ... p*=[1.211493, 6.080599] | L2(m)=2.012e-03 | Linf(m)=6.069e-03 | checkpoint=45 | opt=1.196 s | original total=1.228 s | L2(1000)=9.666e-10 | Linf(1000)=7.157e-09 | extra 1000-refit=0.265 s
[m= 200 | seed  94/100] ... p*=[1.193699, 8.067951] | L2(m)=4.396e-03 | Linf(m)=1.433e-02 | checkpoint=50 | opt=1.058 s | original total=1.084 s | L2(1000)=1.782e-09 | Linf(1000)=7.679e-09 | extra 1000-refit=0.234 s
[m= 200 | seed  95/100] ... p*=[1.098175, 4.336415] | L2(m)=8.039e-04 | Linf(m)=3.007e-03 | checkpoint=8 | opt=1.181 s | original total=1.210 s | L2(1000)=1.876e-09 | Linf(1000)=9.634e-09 | extra 1000-refit=0.252 s
[m= 200 | seed  96/100] ... p*=[1.307362, 5.370820] | L2(m)=2.418e-03 | Linf(m)=8.212e-03 | checkpoint=46 | opt=1.075 s | original total=1.103 s | L2(1000)=9.682e-10 | Linf(1000)=5.022e-09 | extra 1000-refit=0.257 s
[m= 200 | seed  97/100] ... p*=[1.043306, 7.046422] | L2(m)=5.739e-03 | Linf(m)=1.402e-02 | checkpoint=50 | opt=1.070 s | original total=1.099 s | L2(1000)=1.486e-09 | Linf(1000)=7.260e-09 | extra 1000-refit=0.229 s
[m= 200 | seed  98/100] ... p*=[1.259578, 4.694324] | L2(m)=1.813e-03 | Linf(m)=5.781e-03 | checkpoint=45 | opt=1.124 s | original total=1.154 s | L2(1000)=1.407e-09 | Linf(1000)=6.282e-09 | extra 1000-refit=0.241 s
[m= 200 | seed  99/100] ... p*=[1.273261, 4.691659] | L2(m)=7.950e-04 | Linf(m)=3.125e-03 | checkpoint=8 | opt=1.106 s | original total=1.135 s | L2(1000)=9.982e-10 | Linf(1000)=3.880e-09 | extra 1000-refit=0.233 s
[m= 200 | seed 100/100] ... p*=[1.117507, 7.896582] | L2(m)=3.296e-03 | Linf(m)=9.308e-03 | checkpoint=50 | opt=1.055 s | original total=1.082 s | L2(1000)=7.007e-10 | Linf(1000)=3.702e-09 | extra 1000-refit=0.237 s

--------------------------------------------------------------------
Summary for training m = 200
--------------------------------------------------------------------
Original successful runs = 100 / 100
Expanded successful runs = 100 / 100

Original final refit: m = 200

Relative L2
Mean   = 2.592272e-03
Std    = 2.916333e-03
Median = 1.583346e-03
Best   = 1.822907e-04
Worst  = 1.956098e-02

Relative Linf
Mean   = 7.763597e-03
Std    = 7.906323e-03
Median = 4.895818e-03
Best   = 6.572537e-04
Worst  = 4.494956e-02

p1*
Mean = 1.107156
Std  = 0.234813

p2*
Mean = 5.739740
Std  = 1.475362

Mean selected checkpoint = 29.100

Original timing only
Mean basis time        = 0.000419 s
Mean PDE time          = 0.000357 s
Mean cache time        = 0.000494 s
Mean optimization time = 1.150088 s
Mean assembly time     = 0.011040 s
Mean final LS time     = 0.014441 s
Mean test time         = 0.003428 s
Mean original total    = 1.180404 s

Expanded final refit: training m = 200 -> final m = 1000

Expanded relative L2
Mean   = 1.869810e-09
Std    = 1.580194e-09
Median = 1.446471e-09
Best   = 5.049007e-10
Worst  = 1.192510e-08

Expanded relative Linf
Mean   = 8.777751e-09
Std    = 8.165294e-09
Median = 6.509691e-09
Best   = 2.213195e-09
Worst  = 6.269739e-08

Additional m -> 1000 timing only
Mean 1000-basis time    = 0.000499 s
Mean 1000-assembly time = 0.073603 s
Mean 1000-final LS time = 0.155505 s
Mean 1000-test time     = 0.029995 s
Mean extra refit total = 0.259677 s
--------------------------------------------------------------------

####################################################################
Feature number 2 / 5
Training m = 400
Expanded final m = 1000
Initial p = [4.000000, 8.000000]
lambda = 1.000e-07
####################################################################

[m= 400 | seed   1/100] ... p*=[1.648926, 5.518854] | L2(m)=1.931e-06 | Linf(m)=1.372e-05 | checkpoint=19 | opt=2.796 s | original total=2.862 s | L2(1000)=8.226e-10 | Linf(1000)=4.845e-09 | extra 1000-refit=0.239 s
[m= 400 | seed   2/100] ... p*=[1.319692, 5.853102] | L2(m)=1.651e-06 | Linf(m)=8.251e-06 | checkpoint=3 | opt=2.637 s | original total=2.704 s | L2(1000)=6.602e-10 | Linf(1000)=2.929e-09 | extra 1000-refit=0.248 s
[m= 400 | seed   3/100] ... p*=[1.848202, 7.394537] | L2(m)=1.729e-05 | Linf(m)=9.214e-05 | checkpoint=50 | opt=2.710 s | original total=2.781 s | L2(1000)=1.001e-09 | Linf(1000)=5.404e-09 | extra 1000-refit=0.243 s
[m= 400 | seed   4/100] ... p*=[1.377946, 5.316860] | L2(m)=1.432e-06 | Linf(m)=6.962e-06 | checkpoint=4 | opt=2.726 s | original total=2.796 s | L2(1000)=7.887e-10 | Linf(1000)=4.660e-09 | extra 1000-refit=0.254 s
[m= 400 | seed   5/100] ... p*=[1.327750, 5.496269] | L2(m)=5.117e-07 | Linf(m)=2.513e-06 | checkpoint=4 | opt=2.836 s | original total=2.906 s | L2(1000)=8.286e-10 | Linf(1000)=3.786e-09 | extra 1000-refit=0.239 s
[m= 400 | seed   6/100] ... p*=[1.742502, 5.840914] | L2(m)=6.842e-06 | Linf(m)=3.202e-05 | checkpoint=3 | opt=2.708 s | original total=2.786 s | L2(1000)=7.218e-10 | Linf(1000)=3.494e-09 | extra 1000-refit=0.257 s
[m= 400 | seed   7/100] ... p*=[1.310224, 5.746012] | L2(m)=1.132e-06 | Linf(m)=5.731e-06 | checkpoint=3 | opt=2.847 s | original total=2.918 s | L2(1000)=7.197e-10 | Linf(1000)=2.898e-09 | extra 1000-refit=0.259 s
[m= 400 | seed   8/100] ... p*=[1.051434, 4.481712] | L2(m)=8.693e-07 | Linf(m)=4.424e-06 | checkpoint=5 | opt=2.707 s | original total=2.785 s | L2(1000)=2.173e-09 | Linf(1000)=1.364e-08 | extra 1000-refit=0.260 s
[m= 400 | seed   9/100] ... p*=[1.314926, 7.865781] | L2(m)=4.317e-06 | Linf(m)=1.676e-05 | checkpoint=4 | opt=2.769 s | original total=2.832 s | L2(1000)=8.646e-10 | Linf(1000)=3.380e-09 | extra 1000-refit=0.249 s
[m= 400 | seed  10/100] ... p*=[1.306451, 5.707694] | L2(m)=6.081e-07 | Linf(m)=2.568e-06 | checkpoint=26 | opt=2.805 s | original total=2.872 s | L2(1000)=5.918e-10 | Linf(1000)=2.298e-09 | extra 1000-refit=0.235 s
[m= 400 | seed  11/100] ... p*=[1.342035, 5.396152] | L2(m)=6.521e-07 | Linf(m)=3.556e-06 | checkpoint=4 | opt=2.698 s | original total=2.764 s | L2(1000)=6.030e-10 | Linf(1000)=3.925e-09 | extra 1000-refit=0.238 s
[m= 400 | seed  12/100] ... p*=[2.017920, 5.389596] | L2(m)=7.972e-06 | Linf(m)=3.613e-05 | checkpoint=46 | opt=2.752 s | original total=2.814 s | L2(1000)=2.713e-09 | Linf(1000)=1.127e-08 | extra 1000-refit=0.243 s
[m= 400 | seed  13/100] ... p*=[1.246185, 5.314689] | L2(m)=3.839e-07 | Linf(m)=2.513e-06 | checkpoint=4 | opt=2.817 s | original total=2.892 s | L2(1000)=8.523e-10 | Linf(1000)=5.849e-09 | extra 1000-refit=0.245 s
[m= 400 | seed  14/100] ... p*=[1.232189, 5.342655] | L2(m)=6.736e-07 | Linf(m)=2.530e-06 | checkpoint=4 | opt=2.738 s | original total=2.805 s | L2(1000)=7.448e-10 | Linf(1000)=4.851e-09 | extra 1000-refit=0.243 s
[m= 400 | seed  15/100] ... p*=[1.379483, 5.214400] | L2(m)=8.618e-07 | Linf(m)=4.344e-06 | checkpoint=4 | opt=2.927 s | original total=2.997 s | L2(1000)=6.153e-10 | Linf(1000)=3.624e-09 | extra 1000-refit=0.253 s
[m= 400 | seed  16/100] ... p*=[1.342000, 5.357210] | L2(m)=1.053e-06 | Linf(m)=5.797e-06 | checkpoint=4 | opt=2.866 s | original total=2.941 s | L2(1000)=6.391e-10 | Linf(1000)=4.804e-09 | extra 1000-refit=0.279 s
[m= 400 | seed  17/100] ... p*=[1.270889, 5.417791] | L2(m)=4.704e-07 | Linf(m)=2.199e-06 | checkpoint=4 | opt=2.900 s | original total=2.969 s | L2(1000)=9.712e-10 | Linf(1000)=3.930e-09 | extra 1000-refit=0.265 s
[m= 400 | seed  18/100] ... p*=[1.783560, 5.750116] | L2(m)=2.702e-06 | Linf(m)=1.330e-05 | checkpoint=3 | opt=2.810 s | original total=2.876 s | L2(1000)=5.468e-10 | Linf(1000)=2.363e-09 | extra 1000-refit=0.239 s
[m= 400 | seed  19/100] ... p*=[1.776934, 5.768180] | L2(m)=3.618e-06 | Linf(m)=1.583e-05 | checkpoint=3 | opt=2.838 s | original total=2.907 s | L2(1000)=1.153e-09 | Linf(1000)=4.557e-09 | extra 1000-refit=0.266 s
[m= 400 | seed  20/100] ... p*=[1.290796, 10.508005] | L2(m)=3.065e-05 | Linf(m)=1.237e-04 | checkpoint=4 | opt=2.692 s | original total=2.757 s | L2(1000)=7.070e-09 | Linf(1000)=2.328e-08 | extra 1000-refit=0.262 s
[m= 400 | seed  21/100] ... p*=[1.229070, 9.634985] | L2(m)=2.302e-05 | Linf(m)=7.907e-05 | checkpoint=4 | opt=2.688 s | original total=2.750 s | L2(1000)=1.133e-08 | Linf(1000)=2.869e-08 | extra 1000-refit=0.240 s
[m= 400 | seed  22/100] ... p*=[1.259777, 5.792211] | L2(m)=1.390e-06 | Linf(m)=6.557e-06 | checkpoint=3 | opt=2.995 s | original total=3.063 s | L2(1000)=4.615e-10 | Linf(1000)=2.224e-09 | extra 1000-refit=0.266 s
[m= 400 | seed  23/100] ... p*=[1.254230, 10.536870] | L2(m)=7.174e-05 | Linf(m)=2.665e-04 | checkpoint=4 | opt=2.622 s | original total=2.689 s | L2(1000)=6.697e-09 | Linf(1000)=1.544e-08 | extra 1000-refit=0.233 s
[m= 400 | seed  24/100] ... p*=[1.711581, 5.805108] | L2(m)=4.258e-06 | Linf(m)=2.034e-05 | checkpoint=3 | opt=2.690 s | original total=2.761 s | L2(1000)=5.467e-10 | Linf(1000)=3.896e-09 | extra 1000-refit=0.262 s
[m= 400 | seed  25/100] ... p*=[1.447233, 5.302461] | L2(m)=2.466e-06 | Linf(m)=1.507e-05 | checkpoint=30 | opt=2.927 s | original total=2.997 s | L2(1000)=7.291e-10 | Linf(1000)=3.659e-09 | extra 1000-refit=0.239 s
[m= 400 | seed  26/100] ... p*=[1.421962, 5.591980] | L2(m)=1.886e-06 | Linf(m)=8.941e-06 | checkpoint=21 | opt=2.868 s | original total=2.933 s | L2(1000)=7.056e-10 | Linf(1000)=3.799e-09 | extra 1000-refit=0.242 s
[m= 400 | seed  27/100] ... p*=[1.763084, 5.715539] | L2(m)=5.783e-06 | Linf(m)=2.500e-05 | checkpoint=3 | opt=2.865 s | original total=2.933 s | L2(1000)=1.029e-09 | Linf(1000)=4.326e-09 | extra 1000-refit=0.267 s
[m= 400 | seed  28/100] ... p*=[2.001560, 7.308652] | L2(m)=3.180e-05 | Linf(m)=1.138e-04 | checkpoint=50 | opt=2.671 s | original total=2.732 s | L2(1000)=4.598e-10 | Linf(1000)=1.810e-09 | extra 1000-refit=0.229 s
[m= 400 | seed  29/100] ... p*=[1.345067, 5.793133] | L2(m)=7.738e-07 | Linf(m)=2.970e-06 | checkpoint=3 | opt=2.755 s | original total=2.821 s | L2(1000)=4.480e-10 | Linf(1000)=2.489e-09 | extra 1000-refit=0.238 s
[m= 400 | seed  30/100] ... p*=[1.109183, 4.586480] | L2(m)=2.520e-07 | Linf(m)=1.237e-06 | checkpoint=9 | opt=2.745 s | original total=2.824 s | L2(1000)=1.301e-09 | Linf(1000)=6.177e-09 | extra 1000-refit=0.254 s
[m= 400 | seed  31/100] ... p*=[1.202287, 4.973262] | L2(m)=5.327e-07 | Linf(m)=2.051e-06 | checkpoint=50 | opt=2.819 s | original total=2.886 s | L2(1000)=1.213e-09 | Linf(1000)=5.464e-09 | extra 1000-refit=0.252 s
[m= 400 | seed  32/100] ... p*=[1.202777, 5.306211] | L2(m)=3.454e-07 | Linf(m)=1.860e-06 | checkpoint=4 | opt=2.713 s | original total=2.788 s | L2(1000)=8.414e-10 | Linf(1000)=4.860e-09 | extra 1000-refit=0.258 s
[m= 400 | seed  33/100] ... p*=[1.260524, 10.563095] | L2(m)=2.892e-05 | Linf(m)=1.102e-04 | checkpoint=4 | opt=2.634 s | original total=2.694 s | L2(1000)=1.376e-08 | Linf(1000)=3.812e-08 | extra 1000-refit=0.217 s
[m= 400 | seed  34/100] ... p*=[1.182079, 5.383566] | L2(m)=4.771e-07 | Linf(m)=2.074e-06 | checkpoint=4 | opt=2.640 s | original total=2.710 s | L2(1000)=9.618e-10 | Linf(1000)=4.022e-09 | extra 1000-refit=0.243 s
[m= 400 | seed  35/100] ... p*=[1.410608, 5.234756] | L2(m)=9.073e-07 | Linf(m)=3.639e-06 | checkpoint=8 | opt=2.789 s | original total=2.855 s | L2(1000)=9.785e-10 | Linf(1000)=6.539e-09 | extra 1000-refit=0.259 s
[m= 400 | seed  36/100] ... p*=[2.213329, 7.170804] | L2(m)=4.563e-05 | Linf(m)=2.509e-04 | checkpoint=50 | opt=2.650 s | original total=2.717 s | L2(1000)=1.342e-09 | Linf(1000)=5.767e-09 | extra 1000-refit=0.245 s
[m= 400 | seed  37/100] ... p*=[1.439200, 5.486605] | L2(m)=1.630e-06 | Linf(m)=8.562e-06 | checkpoint=40 | opt=2.725 s | original total=2.791 s | L2(1000)=7.773e-10 | Linf(1000)=3.619e-09 | extra 1000-refit=0.253 s
[m= 400 | seed  38/100] ... p*=[1.334013, 5.793255] | L2(m)=6.702e-07 | Linf(m)=3.993e-06 | checkpoint=50 | opt=2.912 s | original total=2.983 s | L2(1000)=1.267e-09 | Linf(1000)=6.428e-09 | extra 1000-refit=0.248 s
[m= 400 | seed  39/100] ... p*=[1.747505, 5.780381] | L2(m)=6.163e-06 | Linf(m)=2.924e-05 | checkpoint=3 | opt=2.760 s | original total=2.825 s | L2(1000)=4.628e-10 | Linf(1000)=1.821e-09 | extra 1000-refit=0.244 s
[m= 400 | seed  40/100] ... p*=[1.382552, 5.196265] | L2(m)=8.271e-07 | Linf(m)=4.121e-06 | checkpoint=4 | opt=2.733 s | original total=2.798 s | L2(1000)=8.413e-10 | Linf(1000)=5.518e-09 | extra 1000-refit=0.243 s
[m= 400 | seed  41/100] ... p*=[1.812625, 9.941324] | L2(m)=5.252e-05 | Linf(m)=2.682e-04 | checkpoint=3 | opt=2.605 s | original total=2.664 s | L2(1000)=8.761e-09 | Linf(1000)=2.310e-08 | extra 1000-refit=0.265 s
[m= 400 | seed  42/100] ... p*=[1.511121, 6.754871] | L2(m)=4.913e-06 | Linf(m)=2.315e-05 | checkpoint=50 | opt=2.781 s | original total=2.849 s | L2(1000)=4.896e-10 | Linf(1000)=2.031e-09 | extra 1000-refit=0.261 s
[m= 400 | seed  43/100] ... p*=[1.281903, 5.415798] | L2(m)=1.042e-06 | Linf(m)=4.833e-06 | checkpoint=4 | opt=2.780 s | original total=2.848 s | L2(1000)=7.079e-10 | Linf(1000)=4.071e-09 | extra 1000-refit=0.240 s
[m= 400 | seed  44/100] ... p*=[1.672363, 5.776747] | L2(m)=3.139e-06 | Linf(m)=1.798e-05 | checkpoint=3 | opt=2.785 s | original total=2.859 s | L2(1000)=5.865e-10 | Linf(1000)=2.605e-09 | extra 1000-refit=0.260 s
[m= 400 | seed  45/100] ... p*=[1.290342, 5.329938] | L2(m)=6.563e-07 | Linf(m)=3.716e-06 | checkpoint=4 | opt=2.942 s | original total=3.014 s | L2(1000)=1.058e-09 | Linf(1000)=4.985e-09 | extra 1000-refit=0.273 s
[m= 400 | seed  46/100] ... p*=[0.931689, 3.730071] | L2(m)=3.891e-07 | Linf(m)=1.372e-06 | checkpoint=10 | opt=2.854 s | original total=2.926 s | L2(1000)=5.782e-09 | Linf(1000)=3.212e-08 | extra 1000-refit=0.249 s
[m= 400 | seed  47/100] ... p*=[1.336928, 5.318068] | L2(m)=9.753e-07 | Linf(m)=5.475e-06 | checkpoint=4 | opt=2.829 s | original total=2.899 s | L2(1000)=9.606e-10 | Linf(1000)=5.159e-09 | extra 1000-refit=0.288 s
[m= 400 | seed  48/100] ... p*=[1.744007, 10.178795] | L2(m)=8.875e-05 | Linf(m)=2.520e-04 | checkpoint=3 | opt=2.690 s | original total=2.751 s | L2(1000)=6.211e-09 | Linf(1000)=2.941e-08 | extra 1000-refit=0.239 s
[m= 400 | seed  49/100] ... p*=[1.301599, 5.353270] | L2(m)=7.063e-07 | Linf(m)=2.788e-06 | checkpoint=4 | opt=2.917 s | original total=2.989 s | L2(1000)=8.970e-10 | Linf(1000)=5.938e-09 | extra 1000-refit=0.232 s
[m= 400 | seed  50/100] ... p*=[1.758684, 5.702295] | L2(m)=2.636e-06 | Linf(m)=1.227e-05 | checkpoint=3 | opt=2.789 s | original total=2.850 s | L2(1000)=6.678e-10 | Linf(1000)=2.653e-09 | extra 1000-refit=0.262 s
[m= 400 | seed  51/100] ... p*=[1.010547, 5.034986] | L2(m)=2.017e-07 | Linf(m)=7.199e-07 | checkpoint=5 | opt=2.739 s | original total=2.813 s | L2(1000)=1.078e-09 | Linf(1000)=7.096e-09 | extra 1000-refit=0.252 s
[m= 400 | seed  52/100] ... p*=[1.327876, 5.360895] | L2(m)=3.016e-06 | Linf(m)=1.595e-05 | checkpoint=4 | opt=2.761 s | original total=2.827 s | L2(1000)=1.101e-09 | Linf(1000)=5.568e-09 | extra 1000-refit=0.261 s
[m= 400 | seed  53/100] ... p*=[1.410463, 5.671724] | L2(m)=9.995e-07 | Linf(m)=5.777e-06 | checkpoint=8 | opt=2.702 s | original total=2.770 s | L2(1000)=9.667e-10 | Linf(1000)=4.721e-09 | extra 1000-refit=0.240 s
[m= 400 | seed  54/100] ... p*=[1.605075, 9.762078] | L2(m)=3.687e-05 | Linf(m)=1.437e-04 | checkpoint=8 | opt=2.659 s | original total=2.722 s | L2(1000)=1.542e-09 | Linf(1000)=4.086e-09 | extra 1000-refit=0.257 s
[m= 400 | seed  55/100] ... p*=[1.709825, 5.760277] | L2(m)=3.148e-06 | Linf(m)=9.352e-06 | checkpoint=3 | opt=2.875 s | original total=2.952 s | L2(1000)=1.109e-09 | Linf(1000)=4.790e-09 | extra 1000-refit=0.257 s
[m= 400 | seed  56/100] ... p*=[1.413623, 7.690551] | L2(m)=9.791e-06 | Linf(m)=3.022e-05 | checkpoint=3 | opt=2.848 s | original total=2.913 s | L2(1000)=6.400e-10 | Linf(1000)=2.457e-09 | extra 1000-refit=0.240 s
[m= 400 | seed  57/100] ... p*=[1.158541, 5.441472] | L2(m)=6.454e-07 | Linf(m)=3.405e-06 | checkpoint=4 | opt=2.730 s | original total=2.797 s | L2(1000)=1.092e-09 | Linf(1000)=6.823e-09 | extra 1000-refit=0.235 s
[m= 400 | seed  58/100] ... p*=[1.317731, 4.835209] | L2(m)=7.004e-07 | Linf(m)=3.930e-06 | checkpoint=9 | opt=2.606 s | original total=2.672 s | L2(1000)=2.133e-09 | Linf(1000)=1.274e-08 | extra 1000-refit=0.263 s
[m= 400 | seed  59/100] ... p*=[1.297528, 5.351716] | L2(m)=5.770e-07 | Linf(m)=3.334e-06 | checkpoint=4 | opt=2.740 s | original total=2.808 s | L2(1000)=8.775e-10 | Linf(1000)=4.284e-09 | extra 1000-refit=0.219 s
[m= 400 | seed  60/100] ... p*=[1.313175, 5.359139] | L2(m)=9.938e-07 | Linf(m)=4.884e-06 | checkpoint=4 | opt=2.817 s | original total=2.888 s | L2(1000)=1.460e-09 | Linf(1000)=9.274e-09 | extra 1000-refit=0.245 s
[m= 400 | seed  61/100] ... p*=[1.652411, 8.702517] | L2(m)=2.963e-05 | Linf(m)=1.351e-04 | checkpoint=8 | opt=2.611 s | original total=2.673 s | L2(1000)=1.421e-09 | Linf(1000)=3.343e-09 | extra 1000-refit=0.229 s
[m= 400 | seed  62/100] ... p*=[1.320769, 10.554985] | L2(m)=4.385e-05 | Linf(m)=2.484e-04 | checkpoint=4 | opt=2.613 s | original total=2.679 s | L2(1000)=4.365e-09 | Linf(1000)=1.561e-08 | extra 1000-refit=0.261 s
[m= 400 | seed  63/100] ... p*=[1.265008, 5.314079] | L2(m)=8.207e-07 | Linf(m)=4.711e-06 | checkpoint=4 | opt=2.712 s | original total=2.778 s | L2(1000)=9.431e-10 | Linf(1000)=4.528e-09 | extra 1000-refit=0.234 s
[m= 400 | seed  64/100] ... p*=[1.390634, 6.663987] | L2(m)=1.742e-06 | Linf(m)=7.090e-06 | checkpoint=50 | opt=2.865 s | original total=2.930 s | L2(1000)=5.905e-10 | Linf(1000)=1.876e-09 | extra 1000-refit=0.232 s
[m= 400 | seed  65/100] ... p*=[1.734432, 5.794821] | L2(m)=1.822e-06 | Linf(m)=7.757e-06 | checkpoint=3 | opt=2.863 s | original total=2.932 s | L2(1000)=8.895e-10 | Linf(1000)=3.857e-09 | extra 1000-refit=0.254 s
[m= 400 | seed  66/100] ... p*=[1.327800, 5.400614] | L2(m)=7.323e-07 | Linf(m)=2.953e-06 | checkpoint=4 | opt=2.951 s | original total=3.024 s | L2(1000)=1.048e-09 | Linf(1000)=6.836e-09 | extra 1000-refit=0.286 s
[m= 400 | seed  67/100] ... p*=[1.356391, 10.035678] | L2(m)=4.655e-05 | Linf(m)=2.143e-04 | checkpoint=8 | opt=2.770 s | original total=2.839 s | L2(1000)=1.267e-08 | Linf(1000)=4.538e-08 | extra 1000-refit=0.245 s
[m= 400 | seed  68/100] ... p*=[1.324856, 5.448204] | L2(m)=1.428e-06 | Linf(m)=7.612e-06 | checkpoint=4 | opt=2.851 s | original total=2.920 s | L2(1000)=7.420e-10 | Linf(1000)=3.462e-09 | extra 1000-refit=0.276 s
[m= 400 | seed  69/100] ... p*=[1.217284, 5.331589] | L2(m)=3.274e-07 | Linf(m)=1.571e-06 | checkpoint=4 | opt=2.785 s | original total=2.857 s | L2(1000)=8.269e-10 | Linf(1000)=4.114e-09 | extra 1000-refit=0.267 s
[m= 400 | seed  70/100] ... p*=[1.123968, 8.595859] | L2(m)=1.641e-05 | Linf(m)=4.896e-05 | checkpoint=9 | opt=2.722 s | original total=2.791 s | L2(1000)=1.948e-09 | Linf(1000)=7.682e-09 | extra 1000-refit=0.240 s
[m= 400 | seed  71/100] ... p*=[1.706981, 10.076101] | L2(m)=3.089e-05 | Linf(m)=1.615e-04 | checkpoint=3 | opt=2.675 s | original total=2.737 s | L2(1000)=2.129e-09 | Linf(1000)=7.141e-09 | extra 1000-refit=0.253 s
[m= 400 | seed  72/100] ... p*=[1.689662, 8.389871] | L2(m)=1.157e-05 | Linf(m)=5.962e-05 | checkpoint=10 | opt=2.800 s | original total=2.867 s | L2(1000)=7.393e-10 | Linf(1000)=2.805e-09 | extra 1000-refit=0.255 s
[m= 400 | seed  73/100] ... p*=[1.370361, 5.209497] | L2(m)=1.445e-06 | Linf(m)=7.642e-06 | checkpoint=4 | opt=2.819 s | original total=2.891 s | L2(1000)=1.001e-09 | Linf(1000)=4.802e-09 | extra 1000-refit=0.266 s
[m= 400 | seed  74/100] ... p*=[1.682673, 9.238858] | L2(m)=5.289e-05 | Linf(m)=2.313e-04 | checkpoint=50 | opt=2.597 s | original total=2.662 s | L2(1000)=1.588e-09 | Linf(1000)=1.027e-08 | extra 1000-refit=0.236 s
[m= 400 | seed  75/100] ... p*=[1.196726, 5.348183] | L2(m)=5.494e-07 | Linf(m)=2.612e-06 | checkpoint=4 | opt=2.898 s | original total=2.964 s | L2(1000)=7.934e-10 | Linf(1000)=3.654e-09 | extra 1000-refit=0.231 s
[m= 400 | seed  76/100] ... p*=[1.310502, 5.581084] | L2(m)=1.152e-06 | Linf(m)=5.219e-06 | checkpoint=4 | opt=2.734 s | original total=2.798 s | L2(1000)=7.048e-10 | Linf(1000)=3.612e-09 | extra 1000-refit=0.257 s
[m= 400 | seed  77/100] ... p*=[1.229214, 5.328552] | L2(m)=4.100e-07 | Linf(m)=1.808e-06 | checkpoint=4 | opt=2.801 s | original total=2.869 s | L2(1000)=6.933e-10 | Linf(1000)=2.815e-09 | extra 1000-refit=0.262 s
[m= 400 | seed  78/100] ... p*=[1.382889, 5.898811] | L2(m)=1.732e-06 | Linf(m)=8.855e-06 | checkpoint=26 | opt=2.728 s | original total=2.796 s | L2(1000)=7.949e-10 | Linf(1000)=3.510e-09 | extra 1000-refit=0.237 s
[m= 400 | seed  79/100] ... p*=[1.210976, 10.450195] | L2(m)=4.889e-05 | Linf(m)=1.650e-04 | checkpoint=4 | opt=2.601 s | original total=2.666 s | L2(1000)=8.019e-09 | Linf(1000)=1.719e-08 | extra 1000-refit=0.234 s
[m= 400 | seed  80/100] ... p*=[1.357045, 5.024226] | L2(m)=9.004e-07 | Linf(m)=5.132e-06 | checkpoint=4 | opt=2.660 s | original total=2.728 s | L2(1000)=9.473e-10 | Linf(1000)=6.555e-09 | extra 1000-refit=0.256 s
[m= 400 | seed  81/100] ... p*=[1.255380, 5.355893] | L2(m)=1.582e-06 | Linf(m)=6.846e-06 | checkpoint=4 | opt=2.785 s | original total=2.857 s | L2(1000)=1.500e-09 | Linf(1000)=6.652e-09 | extra 1000-refit=0.251 s
[m= 400 | seed  82/100] ... p*=[1.324789, 9.378034] | L2(m)=3.551e-05 | Linf(m)=1.059e-04 | checkpoint=9 | opt=2.695 s | original total=2.761 s | L2(1000)=4.245e-09 | Linf(1000)=1.186e-08 | extra 1000-refit=0.274 s
[m= 400 | seed  83/100] ... p*=[1.153608, 9.449340] | L2(m)=1.178e-05 | Linf(m)=5.597e-05 | checkpoint=11 | opt=2.827 s | original total=2.899 s | L2(1000)=4.862e-09 | Linf(1000)=1.443e-08 | extra 1000-refit=0.238 s
[m= 400 | seed  84/100] ... p*=[1.238271, 5.407341] | L2(m)=4.946e-07 | Linf(m)=1.960e-06 | checkpoint=4 | opt=2.834 s | original total=2.907 s | L2(1000)=1.093e-09 | Linf(1000)=6.601e-09 | extra 1000-refit=0.235 s
[m= 400 | seed  85/100] ... p*=[1.306537, 6.151197] | L2(m)=1.470e-06 | Linf(m)=6.139e-06 | checkpoint=27 | opt=3.053 s | original total=3.124 s | L2(1000)=4.253e-10 | Linf(1000)=2.941e-09 | extra 1000-refit=0.259 s
[m= 400 | seed  86/100] ... p*=[1.384482, 4.765882] | L2(m)=5.780e-07 | Linf(m)=2.629e-06 | checkpoint=4 | opt=2.804 s | original total=2.871 s | L2(1000)=1.713e-09 | Linf(1000)=8.725e-09 | extra 1000-refit=0.260 s
[m= 400 | seed  87/100] ... p*=[1.087257, 5.443511] | L2(m)=3.857e-07 | Linf(m)=1.522e-06 | checkpoint=4 | opt=2.854 s | original total=2.927 s | L2(1000)=1.141e-09 | Linf(1000)=5.992e-09 | extra 1000-refit=0.259 s
[m= 400 | seed  88/100] ... p*=[0.998987, 5.027206] | L2(m)=1.837e-07 | Linf(m)=8.489e-07 | checkpoint=5 | opt=2.861 s | original total=2.937 s | L2(1000)=1.072e-09 | Linf(1000)=4.967e-09 | extra 1000-refit=0.272 s
[m= 400 | seed  89/100] ... p*=[1.263345, 5.444753] | L2(m)=4.137e-07 | Linf(m)=2.316e-06 | checkpoint=4 | opt=2.868 s | original total=2.936 s | L2(1000)=8.124e-10 | Linf(1000)=3.820e-09 | extra 1000-refit=0.247 s
[m= 400 | seed  90/100] ... p*=[1.160578, 11.394915] | L2(m)=2.675e-05 | Linf(m)=1.048e-04 | checkpoint=10 | opt=2.680 s | original total=2.774 s | L2(1000)=2.021e-08 | Linf(1000)=5.997e-08 | extra 1000-refit=0.303 s
[m= 400 | seed  91/100] ... p*=[1.243956, 10.147419] | L2(m)=2.419e-05 | Linf(m)=1.161e-04 | checkpoint=4 | opt=3.234 s | original total=3.318 s | L2(1000)=1.888e-08 | Linf(1000)=3.573e-08 | extra 1000-refit=0.320 s
[m= 400 | seed  92/100] ... p*=[1.359074, 4.682291] | L2(m)=7.300e-07 | Linf(m)=3.223e-06 | checkpoint=4 | opt=3.309 s | original total=3.396 s | L2(1000)=1.315e-09 | Linf(1000)=7.584e-09 | extra 1000-refit=0.281 s
[m= 400 | seed  93/100] ... p*=[1.261791, 5.352837] | L2(m)=6.497e-07 | Linf(m)=3.612e-06 | checkpoint=4 | opt=2.789 s | original total=2.868 s | L2(1000)=8.352e-10 | Linf(1000)=3.972e-09 | extra 1000-refit=0.259 s
[m= 400 | seed  94/100] ... p*=[1.231769, 7.917258] | L2(m)=8.039e-06 | Linf(m)=3.290e-05 | checkpoint=50 | opt=2.850 s | original total=2.920 s | L2(1000)=1.025e-09 | Linf(1000)=6.879e-09 | extra 1000-refit=0.249 s
[m= 400 | seed  95/100] ... p*=[1.648676, 5.822399] | L2(m)=2.913e-06 | Linf(m)=1.804e-05 | checkpoint=3 | opt=2.896 s | original total=2.961 s | L2(1000)=5.250e-10 | Linf(1000)=2.475e-09 | extra 1000-refit=0.235 s
[m= 400 | seed  96/100] ... p*=[1.788607, 5.793910] | L2(m)=3.505e-06 | Linf(m)=1.637e-05 | checkpoint=3 | opt=2.823 s | original total=2.893 s | L2(1000)=6.595e-10 | Linf(1000)=2.865e-09 | extra 1000-refit=0.279 s
[m= 400 | seed  97/100] ... p*=[1.348821, 8.135322] | L2(m)=1.619e-05 | Linf(m)=8.838e-05 | checkpoint=9 | opt=2.750 s | original total=2.816 s | L2(1000)=1.921e-09 | Linf(1000)=5.922e-09 | extra 1000-refit=0.264 s
[m= 400 | seed  98/100] ... p*=[1.879507, 6.152741] | L2(m)=5.880e-06 | Linf(m)=3.105e-05 | checkpoint=38 | opt=3.001 s | original total=3.072 s | L2(1000)=6.860e-10 | Linf(1000)=3.641e-09 | extra 1000-refit=0.256 s
[m= 400 | seed  99/100] ... p*=[1.745523, 10.144400] | L2(m)=1.279e-04 | Linf(m)=6.175e-04 | checkpoint=3 | opt=2.796 s | original total=2.858 s | L2(1000)=4.112e-09 | Linf(1000)=1.269e-08 | extra 1000-refit=0.241 s
[m= 400 | seed 100/100] ... p*=[2.157305, 5.477293] | L2(m)=1.240e-05 | Linf(m)=4.874e-05 | checkpoint=42 | opt=2.805 s | original total=2.866 s | L2(1000)=1.786e-09 | Linf(1000)=6.657e-09 | extra 1000-refit=0.237 s

--------------------------------------------------------------------
Summary for training m = 400
--------------------------------------------------------------------
Original successful runs = 100 / 100
Expanded successful runs = 100 / 100

Original final refit: m = 400

Relative L2
Mean   = 1.103081e-05
Std    = 2.065701e-05
Median = 1.640389e-06
Best   = 1.836933e-07
Worst  = 1.279368e-04

Relative Linf
Mean   = 4.756018e-05
Std    = 8.985616e-05
Median = 7.699516e-06
Best   = 7.199483e-07
Worst  = 6.174887e-04

p1*
Mean = 1.417422
Std  = 0.255414

p2*
Mean = 6.492370
Std  = 1.862760

Mean selected checkpoint = 11.410

Original timing only
Mean basis time        = 0.000423 s
Mean PDE time          = 0.000248 s
Mean cache time        = 0.000758 s
Mean optimization time = 2.788021 s
Mean assembly time     = 0.022710 s
Mean final LS time     = 0.038016 s
Mean test time         = 0.006599 s
Mean original total    = 2.856853 s

Expanded final refit: training m = 400 -> final m = 1000

Expanded relative L2
Mean   = 2.210257e-09
Std    = 3.538911e-09
Median = 9.642547e-10
Best   = 4.252786e-10
Worst  = 2.021099e-08

Expanded relative Linf
Mean   = 8.177698e-09
Std    = 9.665383e-09
Median = 4.824475e-09
Best   = 1.810215e-09
Worst  = 5.997108e-08

Additional m -> 1000 timing only
Mean 1000-basis time    = 0.000509 s
Mean 1000-assembly time = 0.069908 s
Mean 1000-final LS time = 0.151898 s
Mean 1000-test time     = 0.029839 s
Mean extra refit total = 0.252176 s
--------------------------------------------------------------------

####################################################################
Feature number 3 / 5
Training m = 600
Expanded final m = 1000
Initial p = [4.000000, 8.000000]
lambda = 1.000e-07
####################################################################

[m= 600 | seed   1/100] ... p*=[1.904809, 7.505700] | L2(m)=8.303e-08 | Linf(m)=4.313e-07 | checkpoint=50 | opt=4.718 s | original total=4.873 s | L2(1000)=1.008e-09 | Linf(1000)=4.126e-09 | extra 1000-refit=0.258 s
[m= 600 | seed   2/100] ... p*=[1.725806, 7.713636] | L2(m)=5.905e-08 | Linf(m)=4.060e-07 | checkpoint=10 | opt=5.056 s | original total=5.179 s | L2(1000)=9.502e-10 | Linf(1000)=1.907e-09 | extra 1000-refit=0.274 s
[m= 600 | seed   3/100] ... p*=[1.707572, 10.045544] | L2(m)=1.418e-07 | Linf(m)=6.428e-07 | checkpoint=3 | opt=4.619 s | original total=4.742 s | L2(1000)=3.157e-09 | Linf(1000)=1.005e-08 | extra 1000-refit=0.243 s
[m= 600 | seed   4/100] ... p*=[1.238806, 8.543447] | L2(m)=3.954e-08 | Linf(m)=1.648e-07 | checkpoint=4 | opt=4.828 s | original total=4.956 s | L2(1000)=2.865e-09 | Linf(1000)=7.019e-09 | extra 1000-refit=0.246 s
[m= 600 | seed   5/100] ... p*=[1.944557, 6.358373] | L2(m)=7.859e-08 | Linf(m)=4.810e-07 | checkpoint=9 | opt=4.860 s | original total=4.981 s | L2(1000)=9.916e-10 | Linf(1000)=4.290e-09 | extra 1000-refit=0.267 s
[m= 600 | seed   6/100] ... p*=[1.342222, 7.152451] | L2(m)=2.158e-08 | Linf(m)=1.244e-07 | checkpoint=8 | opt=5.653 s | original total=5.781 s | L2(1000)=5.655e-10 | Linf(1000)=2.592e-09 | extra 1000-refit=0.249 s
[m= 600 | seed   7/100] ... p*=[1.373900, 7.955246] | L2(m)=2.183e-08 | Linf(m)=9.250e-08 | checkpoint=38 | opt=4.878 s | original total=4.990 s | L2(1000)=2.686e-09 | Linf(1000)=7.612e-09 | extra 1000-refit=0.263 s
[m= 600 | seed   8/100] ... p*=[1.227057, 8.398392] | L2(m)=3.395e-08 | Linf(m)=1.611e-07 | checkpoint=19 | opt=4.809 s | original total=4.939 s | L2(1000)=1.231e-09 | Linf(1000)=3.770e-09 | extra 1000-refit=0.243 s
[m= 600 | seed   9/100] ... p*=[1.753132, 7.079072] | L2(m)=4.511e-08 | Linf(m)=2.259e-07 | checkpoint=48 | opt=4.908 s | original total=5.033 s | L2(1000)=7.073e-10 | Linf(1000)=3.835e-09 | extra 1000-refit=0.267 s
[m= 600 | seed  10/100] ... p*=[1.371839, 10.398668] | L2(m)=4.039e-07 | Linf(m)=2.158e-06 | checkpoint=4 | opt=4.929 s | original total=5.056 s | L2(1000)=1.086e-08 | Linf(1000)=3.132e-08 | extra 1000-refit=0.239 s
[m= 600 | seed  11/100] ... p*=[1.566860, 9.554597] | L2(m)=2.322e-07 | Linf(m)=1.138e-06 | checkpoint=37 | opt=4.963 s | original total=5.075 s | L2(1000)=1.980e-09 | Linf(1000)=6.078e-09 | extra 1000-refit=0.249 s
[m= 600 | seed  12/100] ... p*=[2.119061, 9.045625] | L2(m)=4.768e-07 | Linf(m)=2.724e-06 | checkpoint=28 | opt=4.906 s | original total=5.016 s | L2(1000)=1.304e-09 | Linf(1000)=4.285e-09 | extra 1000-refit=0.254 s
[m= 600 | seed  13/100] ... p*=[1.837219, 8.795721] | L2(m)=2.556e-07 | Linf(m)=1.254e-06 | checkpoint=50 | opt=5.003 s | original total=5.112 s | L2(1000)=1.385e-09 | Linf(1000)=4.427e-09 | extra 1000-refit=0.253 s
[m= 600 | seed  14/100] ... p*=[1.716822, 8.031448] | L2(m)=1.075e-07 | Linf(m)=5.100e-07 | checkpoint=37 | opt=5.023 s | original total=5.146 s | L2(1000)=7.838e-10 | Linf(1000)=2.817e-09 | extra 1000-refit=0.255 s
[m= 600 | seed  15/100] ... p*=[1.763421, 6.957861] | L2(m)=3.119e-08 | Linf(m)=1.764e-07 | checkpoint=50 | opt=4.804 s | original total=4.925 s | L2(1000)=5.282e-10 | Linf(1000)=1.773e-09 | extra 1000-refit=0.265 s
[m= 600 | seed  16/100] ... p*=[1.345532, 8.425844] | L2(m)=5.859e-08 | Linf(m)=3.225e-07 | checkpoint=44 | opt=4.985 s | original total=5.088 s | L2(1000)=1.286e-09 | Linf(1000)=3.724e-09 | extra 1000-refit=0.245 s
[m= 600 | seed  17/100] ... p*=[1.847096, 7.133494] | L2(m)=7.292e-08 | Linf(m)=4.060e-07 | checkpoint=36 | opt=4.896 s | original total=5.049 s | L2(1000)=1.047e-09 | Linf(1000)=3.512e-09 | extra 1000-refit=0.254 s
[m= 600 | seed  18/100] ... p*=[1.545259, 8.563005] | L2(m)=9.398e-08 | Linf(m)=5.348e-07 | checkpoint=37 | opt=5.036 s | original total=5.171 s | L2(1000)=1.404e-09 | Linf(1000)=5.461e-09 | extra 1000-refit=0.275 s
[m= 600 | seed  19/100] ... p*=[2.414334, 6.824675] | L2(m)=1.217e-07 | Linf(m)=6.510e-07 | checkpoint=50 | opt=4.929 s | original total=5.051 s | L2(1000)=2.455e-09 | Linf(1000)=1.069e-08 | extra 1000-refit=0.240 s
[m= 600 | seed  20/100] ... p*=[1.932616, 10.854320] | L2(m)=2.615e-06 | Linf(m)=1.443e-05 | checkpoint=9 | opt=4.849 s | original total=4.948 s | L2(1000)=3.275e-09 | Linf(1000)=1.139e-08 | extra 1000-refit=0.225 s
[m= 600 | seed  21/100] ... p*=[1.468802, 9.121197] | L2(m)=1.001e-07 | Linf(m)=4.853e-07 | checkpoint=48 | opt=4.918 s | original total=5.033 s | L2(1000)=2.336e-09 | Linf(1000)=6.790e-09 | extra 1000-refit=0.264 s
[m= 600 | seed  22/100] ... p*=[1.621845, 6.330402] | L2(m)=1.308e-07 | Linf(m)=9.038e-07 | checkpoint=7 | opt=5.033 s | original total=5.156 s | L2(1000)=4.373e-10 | Linf(1000)=1.936e-09 | extra 1000-refit=0.253 s
[m= 600 | seed  23/100] ... p*=[1.631443, 6.950087] | L2(m)=2.868e-08 | Linf(m)=1.282e-07 | checkpoint=49 | opt=4.793 s | original total=4.917 s | L2(1000)=7.650e-10 | Linf(1000)=3.612e-09 | extra 1000-refit=0.257 s
[m= 600 | seed  24/100] ... p*=[1.569887, 5.874894] | L2(m)=1.895e-08 | Linf(m)=9.272e-08 | checkpoint=9 | opt=5.158 s | original total=5.278 s | L2(1000)=4.676e-10 | Linf(1000)=2.532e-09 | extra 1000-refit=0.274 s
[m= 600 | seed  25/100] ... p*=[1.462757, 9.327437] | L2(m)=1.157e-07 | Linf(m)=5.088e-07 | checkpoint=37 | opt=5.188 s | original total=5.327 s | L2(1000)=1.983e-09 | Linf(1000)=6.312e-09 | extra 1000-refit=0.270 s
[m= 600 | seed  26/100] ... p*=[1.417304, 7.339317] | L2(m)=3.406e-08 | Linf(m)=1.747e-07 | checkpoint=34 | opt=4.819 s | original total=4.935 s | L2(1000)=5.759e-10 | Linf(1000)=2.535e-09 | extra 1000-refit=0.238 s
[m= 600 | seed  27/100] ... p*=[1.887480, 8.572697] | L2(m)=8.354e-08 | Linf(m)=4.337e-07 | checkpoint=29 | opt=5.058 s | original total=5.179 s | L2(1000)=2.190e-09 | Linf(1000)=6.891e-09 | extra 1000-refit=0.243 s
[m= 600 | seed  28/100] ... p*=[2.224156, 8.425454] | L2(m)=1.012e-06 | Linf(m)=5.801e-06 | checkpoint=50 | opt=4.918 s | original total=5.021 s | L2(1000)=9.409e-10 | Linf(1000)=3.067e-09 | extra 1000-refit=0.232 s
[m= 600 | seed  29/100] ... p*=[1.694355, 7.715932] | L2(m)=4.610e-08 | Linf(m)=2.008e-07 | checkpoint=39 | opt=4.873 s | original total=4.978 s | L2(1000)=3.963e-10 | Linf(1000)=1.483e-09 | extra 1000-refit=0.219 s
[m= 600 | seed  30/100] ... p*=[1.305405, 7.195643] | L2(m)=1.731e-08 | Linf(m)=8.016e-08 | checkpoint=19 | opt=5.079 s | original total=5.211 s | L2(1000)=5.952e-10 | Linf(1000)=2.781e-09 | extra 1000-refit=0.234 s
[m= 600 | seed  31/100] ... p*=[1.466435, 7.117890] | L2(m)=2.606e-08 | Linf(m)=1.286e-07 | checkpoint=49 | opt=4.799 s | original total=4.925 s | L2(1000)=6.066e-10 | Linf(1000)=2.233e-09 | extra 1000-refit=0.248 s
[m= 600 | seed  32/100] ... p*=[1.972461, 6.810446] | L2(m)=6.305e-08 | Linf(m)=3.855e-07 | checkpoint=7 | opt=4.799 s | original total=4.919 s | L2(1000)=6.028e-10 | Linf(1000)=2.475e-09 | extra 1000-refit=0.236 s
[m= 600 | seed  33/100] ... p*=[1.720629, 5.888688] | L2(m)=1.738e-08 | Linf(m)=9.023e-08 | checkpoint=3 | opt=4.826 s | original total=4.934 s | L2(1000)=5.064e-10 | Linf(1000)=2.971e-09 | extra 1000-refit=0.252 s
[m= 600 | seed  34/100] ... p*=[1.696571, 7.477497] | L2(m)=3.225e-08 | Linf(m)=1.767e-07 | checkpoint=3 | opt=4.987 s | original total=5.110 s | L2(1000)=5.509e-10 | Linf(1000)=2.560e-09 | extra 1000-refit=0.243 s
[m= 600 | seed  35/100] ... p*=[1.259424, 6.390649] | L2(m)=6.449e-09 | Linf(m)=3.273e-08 | checkpoint=3 | opt=5.081 s | original total=5.214 s | L2(1000)=9.965e-10 | Linf(1000)=3.604e-09 | extra 1000-refit=0.261 s
[m= 600 | seed  36/100] ... p*=[1.122501, 8.042330] | L2(m)=1.563e-08 | Linf(m)=6.247e-08 | checkpoint=40 | opt=4.931 s | original total=5.039 s | L2(1000)=1.585e-09 | Linf(1000)=4.464e-09 | extra 1000-refit=0.250 s
[m= 600 | seed  37/100] ... p*=[2.309292, 6.309291] | L2(m)=1.004e-07 | Linf(m)=5.933e-07 | checkpoint=2 | opt=5.027 s | original total=5.177 s | L2(1000)=1.342e-09 | Linf(1000)=4.960e-09 | extra 1000-refit=0.265 s
[m= 600 | seed  38/100] ... p*=[1.951858, 6.965746] | L2(m)=4.257e-08 | Linf(m)=2.552e-07 | checkpoint=36 | opt=4.845 s | original total=4.983 s | L2(1000)=9.823e-10 | Linf(1000)=6.117e-09 | extra 1000-refit=0.275 s
[m= 600 | seed  39/100] ... p*=[1.290308, 10.650410] | L2(m)=1.626e-07 | Linf(m)=6.913e-07 | checkpoint=4 | opt=5.747 s | original total=5.899 s | L2(1000)=7.874e-09 | Linf(1000)=2.332e-08 | extra 1000-refit=0.272 s
[m= 600 | seed  40/100] ... p*=[1.620455, 5.834300] | L2(m)=1.450e-08 | Linf(m)=9.317e-08 | checkpoint=3 | opt=5.674 s | original total=5.813 s | L2(1000)=1.208e-09 | Linf(1000)=6.112e-09 | extra 1000-refit=0.314 s
[m= 600 | seed  41/100] ... p*=[1.743019, 10.578938] | L2(m)=5.000e-07 | Linf(m)=2.536e-06 | checkpoint=36 | opt=5.662 s | original total=5.794 s | L2(1000)=3.826e-09 | Linf(1000)=1.376e-08 | extra 1000-refit=0.292 s
[m= 600 | seed  42/100] ... p*=[1.782172, 6.762833] | L2(m)=2.695e-08 | Linf(m)=1.384e-07 | checkpoint=50 | opt=5.219 s | original total=5.334 s | L2(1000)=4.267e-10 | Linf(1000)=1.574e-09 | extra 1000-refit=0.254 s
[m= 600 | seed  43/100] ... p*=[1.597659, 9.978583] | L2(m)=1.740e-07 | Linf(m)=9.532e-07 | checkpoint=47 | opt=4.806 s | original total=4.926 s | L2(1000)=4.395e-09 | Linf(1000)=9.153e-09 | extra 1000-refit=0.240 s
[m= 600 | seed  44/100] ... p*=[1.435637, 6.842174] | L2(m)=3.133e-08 | Linf(m)=1.868e-07 | checkpoint=36 | opt=4.834 s | original total=4.954 s | L2(1000)=4.324e-10 | Linf(1000)=1.854e-09 | extra 1000-refit=0.244 s
[m= 600 | seed  45/100] ... p*=[2.001514, 7.839760] | L2(m)=9.393e-08 | Linf(m)=5.044e-07 | checkpoint=13 | opt=4.951 s | original total=5.071 s | L2(1000)=1.526e-09 | Linf(1000)=4.064e-09 | extra 1000-refit=0.244 s
[m= 600 | seed  46/100] ... p*=[1.346601, 8.954632] | L2(m)=1.188e-07 | Linf(m)=4.999e-07 | checkpoint=50 | opt=4.736 s | original total=4.856 s | L2(1000)=7.834e-10 | Linf(1000)=2.391e-09 | extra 1000-refit=0.239 s
[m= 600 | seed  47/100] ... p*=[2.008670, 7.147890] | L2(m)=3.867e-08 | Linf(m)=2.678e-07 | checkpoint=49 | opt=4.945 s | original total=5.074 s | L2(1000)=5.864e-10 | Linf(1000)=1.745e-09 | extra 1000-refit=0.245 s
[m= 600 | seed  48/100] ... p*=[1.015863, 5.580207] | L2(m)=9.915e-09 | Linf(m)=3.935e-08 | checkpoint=4 | opt=4.898 s | original total=5.022 s | L2(1000)=1.421e-09 | Linf(1000)=1.193e-08 | extra 1000-refit=0.259 s
[m= 600 | seed  49/100] ... p*=[1.792490, 9.136228] | L2(m)=3.740e-07 | Linf(m)=1.971e-06 | checkpoint=34 | opt=5.154 s | original total=5.261 s | L2(1000)=2.611e-09 | Linf(1000)=8.683e-09 | extra 1000-refit=0.263 s
[m= 600 | seed  50/100] ... p*=[1.604190, 9.858447] | L2(m)=1.902e-07 | Linf(m)=8.551e-07 | checkpoint=7 | opt=4.654 s | original total=4.768 s | L2(1000)=2.570e-09 | Linf(1000)=1.057e-08 | extra 1000-refit=0.242 s
[m= 600 | seed  51/100] ... p*=[1.718526, 7.465145] | L2(m)=2.377e-08 | Linf(m)=1.214e-07 | checkpoint=50 | opt=4.804 s | original total=4.928 s | L2(1000)=3.563e-10 | Linf(1000)=1.766e-09 | extra 1000-refit=0.262 s
[m= 600 | seed  52/100] ... p*=[1.704095, 8.157396] | L2(m)=8.408e-08 | Linf(m)=4.742e-07 | checkpoint=41 | opt=4.776 s | original total=4.892 s | L2(1000)=1.065e-09 | Linf(1000)=2.663e-09 | extra 1000-refit=0.251 s
[m= 600 | seed  53/100] ... p*=[1.342762, 8.431150] | L2(m)=2.698e-08 | Linf(m)=1.600e-07 | checkpoint=50 | opt=4.999 s | original total=5.120 s | L2(1000)=2.033e-09 | Linf(1000)=5.447e-09 | extra 1000-refit=0.263 s
[m= 600 | seed  54/100] ... p*=[1.490903, 9.850407] | L2(m)=1.290e-07 | Linf(m)=7.971e-07 | checkpoint=12 | opt=4.712 s | original total=4.829 s | L2(1000)=2.273e-09 | Linf(1000)=5.777e-09 | extra 1000-refit=0.263 s
[m= 600 | seed  55/100] ... p*=[1.747610, 7.466267] | L2(m)=4.183e-08 | Linf(m)=2.045e-07 | checkpoint=25 | opt=5.069 s | original total=5.190 s | L2(1000)=8.005e-10 | Linf(1000)=2.680e-09 | extra 1000-refit=0.246 s
[m= 600 | seed  56/100] ... p*=[1.808847, 8.119547] | L2(m)=1.040e-07 | Linf(m)=5.534e-07 | checkpoint=50 | opt=4.887 s | original total=5.002 s | L2(1000)=1.139e-09 | Linf(1000)=3.608e-09 | extra 1000-refit=0.260 s
[m= 600 | seed  57/100] ... p*=[2.017037, 7.387291] | L2(m)=1.933e-07 | Linf(m)=1.373e-06 | checkpoint=34 | opt=4.825 s | original total=4.940 s | L2(1000)=9.268e-10 | Linf(1000)=3.874e-09 | extra 1000-refit=0.269 s
[m= 600 | seed  58/100] ... p*=[1.985727, 9.908270] | L2(m)=8.800e-07 | Linf(m)=5.108e-06 | checkpoint=9 | opt=4.722 s | original total=4.841 s | L2(1000)=2.196e-09 | Linf(1000)=6.709e-09 | extra 1000-refit=0.242 s
[m= 600 | seed  59/100] ... p*=[1.840389, 8.487632] | L2(m)=1.166e-07 | Linf(m)=5.809e-07 | checkpoint=10 | opt=4.793 s | original total=4.904 s | L2(1000)=9.148e-10 | Linf(1000)=4.495e-09 | extra 1000-refit=0.275 s
[m= 600 | seed  60/100] ... p*=[2.243073, 9.297042] | L2(m)=6.219e-07 | Linf(m)=3.937e-06 | checkpoint=36 | opt=4.752 s | original total=4.855 s | L2(1000)=2.942e-09 | Linf(1000)=7.812e-09 | extra 1000-refit=0.237 s
[m= 600 | seed  61/100] ... p*=[1.318935, 7.408635] | L2(m)=2.180e-08 | Linf(m)=1.298e-07 | checkpoint=25 | opt=4.716 s | original total=4.831 s | L2(1000)=1.385e-09 | Linf(1000)=3.181e-09 | extra 1000-refit=0.225 s
[m= 600 | seed  62/100] ... p*=[1.536667, 6.469892] | L2(m)=1.375e-08 | Linf(m)=6.233e-08 | checkpoint=49 | opt=4.580 s | original total=4.701 s | L2(1000)=9.690e-10 | Linf(1000)=6.236e-09 | extra 1000-refit=0.276 s
[m= 600 | seed  63/100] ... p*=[1.280083, 10.342124] | L2(m)=2.993e-07 | Linf(m)=1.128e-06 | checkpoint=4 | opt=4.650 s | original total=4.773 s | L2(1000)=1.736e-09 | Linf(1000)=5.367e-09 | extra 1000-refit=0.222 s
[m= 600 | seed  64/100] ... p*=[2.257333, 9.476059] | L2(m)=9.497e-07 | Linf(m)=7.441e-06 | checkpoint=30 | opt=4.673 s | original total=4.781 s | L2(1000)=1.670e-09 | Linf(1000)=4.130e-09 | extra 1000-refit=0.246 s
[m= 600 | seed  65/100] ... p*=[1.734492, 6.057665] | L2(m)=9.247e-09 | Linf(m)=5.450e-08 | checkpoint=3 | opt=4.824 s | original total=4.938 s | L2(1000)=5.129e-10 | Linf(1000)=1.826e-09 | extra 1000-refit=0.280 s
[m= 600 | seed  66/100] ... p*=[1.862378, 6.587362] | L2(m)=2.485e-08 | Linf(m)=1.463e-07 | checkpoint=8 | opt=4.597 s | original total=4.725 s | L2(1000)=5.715e-10 | Linf(1000)=2.261e-09 | extra 1000-refit=0.252 s
[m= 600 | seed  67/100] ... p*=[1.785890, 8.802318] | L2(m)=3.027e-07 | Linf(m)=1.452e-06 | checkpoint=47 | opt=4.738 s | original total=4.853 s | L2(1000)=2.238e-09 | Linf(1000)=1.054e-08 | extra 1000-refit=0.238 s
[m= 600 | seed  68/100] ... p*=[2.458589, 6.915069] | L2(m)=1.771e-07 | Linf(m)=1.016e-06 | checkpoint=49 | opt=4.893 s | original total=4.993 s | L2(1000)=1.860e-09 | Linf(1000)=7.136e-09 | extra 1000-refit=0.249 s
[m= 600 | seed  69/100] ... p*=[2.242962, 9.641046] | L2(m)=6.807e-07 | Linf(m)=3.284e-06 | checkpoint=2 | opt=4.787 s | original total=4.902 s | L2(1000)=1.924e-09 | Linf(1000)=1.206e-08 | extra 1000-refit=0.232 s
[m= 600 | seed  70/100] ... p*=[1.600727, 5.407753] | L2(m)=1.648e-08 | Linf(m)=1.176e-07 | checkpoint=7 | opt=4.809 s | original total=4.944 s | L2(1000)=8.776e-10 | Linf(1000)=4.458e-09 | extra 1000-refit=0.254 s
[m= 600 | seed  71/100] ... p*=[1.665344, 6.187379] | L2(m)=1.371e-08 | Linf(m)=7.743e-08 | checkpoint=6 | opt=4.720 s | original total=4.834 s | L2(1000)=4.641e-10 | Linf(1000)=2.520e-09 | extra 1000-refit=0.246 s
[m= 600 | seed  72/100] ... p*=[2.160461, 9.578118] | L2(m)=5.662e-07 | Linf(m)=2.916e-06 | checkpoint=22 | opt=4.862 s | original total=4.964 s | L2(1000)=1.127e-09 | Linf(1000)=3.389e-09 | extra 1000-refit=0.235 s
[m= 600 | seed  73/100] ... p*=[2.059603, 8.762907] | L2(m)=1.019e-06 | Linf(m)=5.855e-06 | checkpoint=50 | opt=4.728 s | original total=4.841 s | L2(1000)=1.363e-09 | Linf(1000)=9.055e-09 | extra 1000-refit=0.257 s
[m= 600 | seed  74/100] ... p*=[1.528394, 6.552375] | L2(m)=1.439e-08 | Linf(m)=7.189e-08 | checkpoint=8 | opt=4.731 s | original total=4.861 s | L2(1000)=8.035e-10 | Linf(1000)=3.831e-09 | extra 1000-refit=0.249 s
[m= 600 | seed  75/100] ... p*=[1.323872, 10.451440] | L2(m)=1.301e-07 | Linf(m)=6.232e-07 | checkpoint=31 | opt=4.809 s | original total=4.930 s | L2(1000)=1.870e-09 | Linf(1000)=7.669e-09 | extra 1000-refit=0.240 s
[m= 600 | seed  76/100] ... p*=[1.320246, 5.439576] | L2(m)=1.575e-08 | Linf(m)=1.094e-07 | checkpoint=4 | opt=4.780 s | original total=4.890 s | L2(1000)=7.138e-10 | Linf(1000)=4.125e-09 | extra 1000-refit=0.269 s
[m= 600 | seed  77/100] ... p*=[1.691660, 8.700361] | L2(m)=8.864e-08 | Linf(m)=4.889e-07 | checkpoint=3 | opt=4.644 s | original total=4.767 s | L2(1000)=8.700e-10 | Linf(1000)=2.671e-09 | extra 1000-refit=0.238 s
[m= 600 | seed  78/100] ... p*=[1.356733, 6.873048] | L2(m)=1.390e-08 | Linf(m)=8.113e-08 | checkpoint=38 | opt=4.701 s | original total=4.827 s | L2(1000)=6.375e-10 | Linf(1000)=3.455e-09 | extra 1000-refit=0.256 s
[m= 600 | seed  79/100] ... p*=[2.106704, 7.708087] | L2(m)=1.945e-07 | Linf(m)=1.344e-06 | checkpoint=32 | opt=4.755 s | original total=4.882 s | L2(1000)=1.122e-09 | Linf(1000)=4.520e-09 | extra 1000-refit=0.215 s
[m= 600 | seed  80/100] ... p*=[1.397240, 9.212065] | L2(m)=9.291e-08 | Linf(m)=4.553e-07 | checkpoint=50 | opt=4.665 s | original total=4.786 s | L2(1000)=1.439e-09 | Linf(1000)=3.538e-09 | extra 1000-refit=0.251 s
[m= 600 | seed  81/100] ... p*=[1.527841, 5.807099] | L2(m)=2.766e-08 | Linf(m)=1.210e-07 | checkpoint=3 | opt=4.781 s | original total=4.919 s | L2(1000)=1.596e-09 | Linf(1000)=5.404e-09 | extra 1000-refit=0.245 s
[m= 600 | seed  82/100] ... p*=[1.442272, 10.717587] | L2(m)=3.275e-07 | Linf(m)=1.869e-06 | checkpoint=8 | opt=4.482 s | original total=4.603 s | L2(1000)=4.976e-09 | Linf(1000)=1.530e-08 | extra 1000-refit=0.228 s
[m= 600 | seed  83/100] ... p*=[1.892746, 8.338888] | L2(m)=1.335e-07 | Linf(m)=1.104e-06 | checkpoint=46 | opt=4.676 s | original total=4.796 s | L2(1000)=6.027e-10 | Linf(1000)=2.685e-09 | extra 1000-refit=0.242 s
[m= 600 | seed  84/100] ... p*=[1.758957, 8.149611] | L2(m)=1.466e-07 | Linf(m)=9.784e-07 | checkpoint=37 | opt=4.702 s | original total=4.807 s | L2(1000)=8.769e-10 | Linf(1000)=2.741e-09 | extra 1000-refit=0.247 s
[m= 600 | seed  85/100] ... p*=[2.274560, 6.819167] | L2(m)=9.063e-08 | Linf(m)=3.527e-07 | checkpoint=42 | opt=4.654 s | original total=4.774 s | L2(1000)=1.080e-09 | Linf(1000)=3.388e-09 | extra 1000-refit=0.234 s
[m= 600 | seed  86/100] ... p*=[2.167853, 5.148589] | L2(m)=9.302e-08 | Linf(m)=6.526e-07 | checkpoint=13 | opt=4.692 s | original total=4.811 s | L2(1000)=3.286e-09 | Linf(1000)=8.995e-09 | extra 1000-refit=0.224 s
[m= 600 | seed  87/100] ... p*=[1.627500, 5.916722] | L2(m)=2.549e-08 | Linf(m)=1.328e-07 | checkpoint=3 | opt=4.650 s | original total=4.772 s | L2(1000)=8.395e-10 | Linf(1000)=4.058e-09 | extra 1000-refit=0.257 s
[m= 600 | seed  88/100] ... p*=[1.469615, 7.524594] | L2(m)=2.716e-08 | Linf(m)=1.460e-07 | checkpoint=43 | opt=4.749 s | original total=4.860 s | L2(1000)=9.464e-10 | Linf(1000)=2.948e-09 | extra 1000-refit=0.251 s
[m= 600 | seed  89/100] ... p*=[1.668736, 10.179792] | L2(m)=2.238e-07 | Linf(m)=1.281e-06 | checkpoint=8 | opt=4.662 s | original total=4.788 s | L2(1000)=2.242e-09 | Linf(1000)=5.016e-09 | extra 1000-refit=0.218 s
[m= 600 | seed  90/100] ... p*=[1.656839, 7.755523] | L2(m)=5.123e-08 | Linf(m)=3.493e-07 | checkpoint=50 | opt=4.908 s | original total=5.027 s | L2(1000)=6.647e-10 | Linf(1000)=2.370e-09 | extra 1000-refit=0.251 s
[m= 600 | seed  91/100] ... p*=[1.375171, 10.616249] | L2(m)=3.119e-07 | Linf(m)=1.332e-06 | checkpoint=4 | opt=4.606 s | original total=4.731 s | L2(1000)=1.485e-08 | Linf(1000)=2.166e-08 | extra 1000-refit=0.244 s
[m= 600 | seed  92/100] ... p*=[1.770675, 6.593367] | L2(m)=4.402e-08 | Linf(m)=2.231e-07 | checkpoint=49 | opt=4.654 s | original total=4.763 s | L2(1000)=5.787e-10 | Linf(1000)=2.263e-09 | extra 1000-refit=0.245 s
[m= 600 | seed  93/100] ... p*=[1.728692, 7.477104] | L2(m)=4.118e-08 | Linf(m)=2.326e-07 | checkpoint=36 | opt=4.755 s | original total=4.906 s | L2(1000)=6.050e-10 | Linf(1000)=1.909e-09 | extra 1000-refit=0.271 s
[m= 600 | seed  94/100] ... p*=[1.472622, 6.910168] | L2(m)=1.844e-08 | Linf(m)=7.389e-08 | checkpoint=50 | opt=4.950 s | original total=5.084 s | L2(1000)=4.123e-10 | Linf(1000)=2.455e-09 | extra 1000-refit=0.265 s
[m= 600 | seed  95/100] ... p*=[2.053751, 7.630747] | L2(m)=1.920e-07 | Linf(m)=1.273e-06 | checkpoint=28 | opt=4.726 s | original total=4.843 s | L2(1000)=1.087e-09 | Linf(1000)=2.722e-09 | extra 1000-refit=0.262 s
[m= 600 | seed  96/100] ... p*=[2.445749, 8.008548] | L2(m)=7.167e-07 | Linf(m)=3.780e-06 | checkpoint=33 | opt=4.710 s | original total=4.834 s | L2(1000)=1.270e-09 | Linf(1000)=6.195e-09 | extra 1000-refit=0.236 s
[m= 600 | seed  97/100] ... p*=[1.257151, 10.566010] | L2(m)=2.668e-07 | Linf(m)=1.382e-06 | checkpoint=4 | opt=4.554 s | original total=4.668 s | L2(1000)=3.273e-09 | Linf(1000)=1.099e-08 | extra 1000-refit=0.248 s
[m= 600 | seed  98/100] ... p*=[1.424078, 7.369462] | L2(m)=3.418e-08 | Linf(m)=1.914e-07 | checkpoint=29 | opt=4.826 s | original total=4.955 s | L2(1000)=7.818e-10 | Linf(1000)=3.046e-09 | extra 1000-refit=0.269 s
[m= 600 | seed  99/100] ... p*=[1.756430, 10.193071] | L2(m)=6.768e-07 | Linf(m)=3.942e-06 | checkpoint=3 | opt=4.458 s | original total=4.569 s | L2(1000)=3.319e-09 | Linf(1000)=1.107e-08 | extra 1000-refit=0.220 s
[m= 600 | seed 100/100] ... p*=[1.245654, 7.084701] | L2(m)=1.323e-08 | Linf(m)=4.880e-08 | checkpoint=39 | opt=5.036 s | original total=5.148 s | L2(1000)=5.368e-10 | Linf(1000)=1.948e-09 | extra 1000-refit=0.243 s

--------------------------------------------------------------------
Summary for training m = 600
--------------------------------------------------------------------
Original successful runs = 100 / 100
Expanded successful runs = 100 / 100

Original final refit: m = 600

Relative L2
Mean   = 1.894722e-07
Std    = 3.349678e-07
Median = 8.381088e-08
Best   = 6.448936e-09
Worst  = 2.614834e-06

Relative Linf
Mean   = 1.059357e-06
Std    = 1.925036e-06
Median = 4.444891e-07
Best   = 3.272800e-08
Worst  = 1.443460e-05

p1*
Mean = 1.693033
Std  = 0.318260

p2*
Mean = 7.963075
Std  = 1.459701

Mean selected checkpoint = 26.810

Original timing only
Mean basis time        = 0.000439 s
Mean PDE time          = 0.000244 s
Mean cache time        = 0.001049 s
Mean optimization time = 4.862958 s
Mean assembly time     = 0.038976 s
Mean final LS time     = 0.068986 s
Mean test time         = 0.011258 s
Mean original total    = 4.983992 s

Expanded final refit: training m = 600 -> final m = 1000

Expanded relative L2
Mean   = 1.686121e-09
Std    = 1.989340e-09
Median = 1.083371e-09
Best   = 3.563037e-10
Worst  = 1.484745e-08

Expanded relative Linf
Mean   = 5.571299e-09
Std    = 4.698025e-09
Median = 4.061363e-09
Best   = 1.483206e-09
Worst  = 3.132217e-08

Additional m -> 1000 timing only
Mean 1000-basis time    = 0.000518 s
Mean 1000-assembly time = 0.070311 s
Mean 1000-final LS time = 0.150083 s
Mean 1000-test time     = 0.029911 s
Mean extra refit total = 0.250845 s
--------------------------------------------------------------------

####################################################################
Feature number 4 / 5
Training m = 800
Expanded final m = 1000
Initial p = [4.000000, 8.000000]
lambda = 1.000e-07
####################################################################

[m= 800 | seed   1/100] ... p*=[1.975847, 10.659062] | L2(m)=2.252e-08 | Linf(m)=1.392e-07 | checkpoint=46 | opt=8.654 s | original total=8.827 s | L2(1000)=9.529e-09 | Linf(1000)=2.594e-08 | extra 1000-refit=0.234 s
[m= 800 | seed   2/100] ... p*=[1.686651, 7.664164] | L2(m)=1.645e-09 | Linf(m)=5.693e-09 | checkpoint=5 | opt=7.900 s | original total=8.069 s | L2(1000)=1.269e-09 | Linf(1000)=2.299e-09 | extra 1000-refit=0.233 s
[m= 800 | seed   3/100] ... p*=[1.588888, 8.931844] | L2(m)=3.129e-09 | Linf(m)=1.074e-08 | checkpoint=39 | opt=7.440 s | original total=7.625 s | L2(1000)=2.589e-09 | Linf(1000)=7.001e-09 | extra 1000-refit=0.232 s
[m= 800 | seed   4/100] ... p*=[1.759107, 9.634336] | L2(m)=4.180e-09 | Linf(m)=2.137e-08 | checkpoint=11 | opt=7.930 s | original total=8.100 s | L2(1000)=8.477e-09 | Linf(1000)=1.561e-08 | extra 1000-refit=0.250 s
[m= 800 | seed   5/100] ... p*=[1.589177, 9.422635] | L2(m)=6.974e-09 | Linf(m)=2.353e-08 | checkpoint=36 | opt=8.022 s | original total=8.217 s | L2(1000)=2.484e-09 | Linf(1000)=7.750e-09 | extra 1000-refit=0.271 s
[m= 800 | seed   6/100] ... p*=[1.670465, 10.264186] | L2(m)=1.314e-08 | Linf(m)=3.702e-08 | checkpoint=3 | opt=7.708 s | original total=7.876 s | L2(1000)=3.960e-09 | Linf(1000)=1.194e-08 | extra 1000-refit=0.232 s
[m= 800 | seed   7/100] ... p*=[1.952536, 7.262299] | L2(m)=3.532e-09 | Linf(m)=1.341e-08 | checkpoint=49 | opt=7.278 s | original total=7.445 s | L2(1000)=9.150e-10 | Linf(1000)=3.738e-09 | extra 1000-refit=0.238 s
[m= 800 | seed   8/100] ... p*=[2.184715, 10.623161] | L2(m)=2.225e-08 | Linf(m)=1.319e-07 | checkpoint=30 | opt=7.826 s | original total=7.982 s | L2(1000)=2.372e-09 | Linf(1000)=8.997e-09 | extra 1000-refit=0.224 s
[m= 800 | seed   9/100] ... p*=[1.796987, 7.641197] | L2(m)=5.450e-09 | Linf(m)=3.990e-08 | checkpoint=39 | opt=7.588 s | original total=7.774 s | L2(1000)=4.116e-10 | Linf(1000)=2.524e-09 | extra 1000-refit=0.247 s
[m= 800 | seed  10/100] ... p*=[2.030836, 9.264022] | L2(m)=4.715e-09 | Linf(m)=2.123e-08 | checkpoint=34 | opt=7.825 s | original total=7.989 s | L2(1000)=1.516e-09 | Linf(1000)=4.425e-09 | extra 1000-refit=0.239 s
[m= 800 | seed  11/100] ... p*=[1.673091, 7.950194] | L2(m)=4.561e-09 | Linf(m)=2.118e-08 | checkpoint=45 | opt=7.835 s | original total=8.013 s | L2(1000)=7.958e-10 | Linf(1000)=2.572e-09 | extra 1000-refit=0.236 s
[m= 800 | seed  12/100] ... p*=[1.972119, 9.724684] | L2(m)=8.483e-09 | Linf(m)=2.966e-08 | checkpoint=50 | opt=7.654 s | original total=7.821 s | L2(1000)=8.188e-10 | Linf(1000)=3.028e-09 | extra 1000-refit=0.242 s
[m= 800 | seed  13/100] ... p*=[1.610668, 9.316028] | L2(m)=4.516e-09 | Linf(m)=2.014e-08 | checkpoint=41 | opt=7.527 s | original total=7.693 s | L2(1000)=2.009e-09 | Linf(1000)=8.846e-09 | extra 1000-refit=0.237 s
[m= 800 | seed  14/100] ... p*=[1.750661, 11.442274] | L2(m)=1.983e-08 | Linf(m)=1.012e-07 | checkpoint=50 | opt=7.941 s | original total=8.112 s | L2(1000)=1.161e-08 | Linf(1000)=3.770e-08 | extra 1000-refit=0.225 s
[m= 800 | seed  15/100] ... p*=[1.966435, 9.900940] | L2(m)=1.166e-08 | Linf(m)=2.975e-08 | checkpoint=44 | opt=7.405 s | original total=7.572 s | L2(1000)=1.374e-09 | Linf(1000)=3.702e-09 | extra 1000-refit=0.235 s
[m= 800 | seed  16/100] ... p*=[1.635234, 9.091637] | L2(m)=4.610e-09 | Linf(m)=1.773e-08 | checkpoint=24 | opt=7.840 s | original total=8.013 s | L2(1000)=1.124e-09 | Linf(1000)=3.371e-09 | extra 1000-refit=0.255 s
[m= 800 | seed  17/100] ... p*=[1.609194, 8.879308] | L2(m)=5.301e-09 | Linf(m)=1.692e-08 | checkpoint=20 | opt=7.472 s | original total=7.650 s | L2(1000)=2.347e-09 | Linf(1000)=7.684e-09 | extra 1000-refit=0.240 s
[m= 800 | seed  18/100] ... p*=[2.704819, 9.065397] | L2(m)=2.453e-08 | Linf(m)=1.500e-07 | checkpoint=37 | opt=7.201 s | original total=7.365 s | L2(1000)=2.019e-09 | Linf(1000)=5.553e-09 | extra 1000-refit=0.236 s
[m= 800 | seed  19/100] ... p*=[1.428874, 9.349440] | L2(m)=5.717e-09 | Linf(m)=2.211e-08 | checkpoint=39 | opt=7.861 s | original total=8.021 s | L2(1000)=3.163e-09 | Linf(1000)=1.076e-08 | extra 1000-refit=0.261 s
[m= 800 | seed  20/100] ... p*=[1.833322, 10.103531] | L2(m)=7.488e-09 | Linf(m)=3.073e-08 | checkpoint=41 | opt=7.409 s | original total=7.568 s | L2(1000)=2.593e-09 | Linf(1000)=7.406e-09 | extra 1000-refit=0.237 s
[m= 800 | seed  21/100] ... p*=[1.441827, 10.143410] | L2(m)=8.238e-09 | Linf(m)=1.977e-08 | checkpoint=3 | opt=7.678 s | original total=7.850 s | L2(1000)=2.542e-09 | Linf(1000)=7.643e-09 | extra 1000-refit=0.248 s
[m= 800 | seed  22/100] ... p*=[2.047260, 9.929776] | L2(m)=1.154e-08 | Linf(m)=6.454e-08 | checkpoint=42 | opt=8.013 s | original total=8.190 s | L2(1000)=3.284e-09 | Linf(1000)=1.205e-08 | extra 1000-refit=0.235 s
[m= 800 | seed  23/100] ... p*=[1.509150, 7.839711] | L2(m)=2.487e-09 | Linf(m)=1.364e-08 | checkpoint=40 | opt=7.481 s | original total=7.647 s | L2(1000)=1.593e-09 | Linf(1000)=5.988e-09 | extra 1000-refit=0.249 s
[m= 800 | seed  24/100] ... p*=[1.677931, 8.910361] | L2(m)=3.222e-09 | Linf(m)=1.198e-08 | checkpoint=40 | opt=7.799 s | original total=8.009 s | L2(1000)=1.116e-09 | Linf(1000)=2.947e-09 | extra 1000-refit=0.233 s
[m= 800 | seed  25/100] ... p*=[1.561723, 11.698460] | L2(m)=2.255e-08 | Linf(m)=9.743e-08 | checkpoint=8 | opt=7.514 s | original total=7.690 s | L2(1000)=1.137e-08 | Linf(1000)=3.440e-08 | extra 1000-refit=0.234 s
[m= 800 | seed  26/100] ... p*=[1.543174, 8.664621] | L2(m)=2.941e-09 | Linf(m)=1.918e-08 | checkpoint=47 | opt=8.093 s | original total=8.276 s | L2(1000)=1.294e-09 | Linf(1000)=4.448e-09 | extra 1000-refit=0.235 s
[m= 800 | seed  27/100] ... p*=[1.827232, 7.893640] | L2(m)=4.785e-09 | Linf(m)=2.126e-08 | checkpoint=33 | opt=7.480 s | original total=7.666 s | L2(1000)=8.873e-10 | Linf(1000)=5.290e-09 | extra 1000-refit=0.219 s
[m= 800 | seed  28/100] ... p*=[1.849995, 8.019303] | L2(m)=5.052e-09 | Linf(m)=1.533e-08 | checkpoint=50 | opt=7.895 s | original total=8.082 s | L2(1000)=5.040e-10 | Linf(1000)=1.964e-09 | extra 1000-refit=0.237 s
[m= 800 | seed  29/100] ... p*=[2.123958, 9.931379] | L2(m)=3.014e-08 | Linf(m)=1.020e-07 | checkpoint=24 | opt=7.440 s | original total=7.600 s | L2(1000)=3.629e-09 | Linf(1000)=8.167e-09 | extra 1000-refit=0.247 s
[m= 800 | seed  30/100] ... p*=[1.988761, 9.701022] | L2(m)=8.276e-09 | Linf(m)=3.559e-08 | checkpoint=44 | opt=7.871 s | original total=8.042 s | L2(1000)=1.840e-09 | Linf(1000)=5.656e-09 | extra 1000-refit=0.235 s
[m= 800 | seed  31/100] ... p*=[1.451367, 8.560468] | L2(m)=3.059e-09 | Linf(m)=1.270e-08 | checkpoint=50 | opt=7.372 s | original total=7.557 s | L2(1000)=1.403e-09 | Linf(1000)=3.807e-09 | extra 1000-refit=0.235 s
[m= 800 | seed  32/100] ... p*=[1.320559, 10.787465] | L2(m)=4.101e-08 | Linf(m)=7.824e-08 | checkpoint=6 | opt=7.647 s | original total=7.821 s | L2(1000)=5.998e-09 | Linf(1000)=1.201e-08 | extra 1000-refit=0.243 s
[m= 800 | seed  33/100] ... p*=[1.669780, 7.656888] | L2(m)=2.593e-09 | Linf(m)=1.176e-08 | checkpoint=40 | opt=7.853 s | original total=8.051 s | L2(1000)=9.562e-10 | Linf(1000)=4.799e-09 | extra 1000-refit=0.244 s
[m= 800 | seed  34/100] ... p*=[1.734299, 7.603115] | L2(m)=2.145e-09 | Linf(m)=1.127e-08 | checkpoint=49 | opt=7.671 s | original total=7.859 s | L2(1000)=6.134e-10 | Linf(1000)=2.276e-09 | extra 1000-refit=0.234 s
[m= 800 | seed  35/100] ... p*=[1.359142, 10.815663] | L2(m)=2.323e-08 | Linf(m)=5.680e-08 | checkpoint=5 | opt=7.388 s | original total=7.551 s | L2(1000)=4.283e-09 | Linf(1000)=1.264e-08 | extra 1000-refit=0.233 s
[m= 800 | seed  36/100] ... p*=[2.113289, 12.059449] | L2(m)=1.415e-07 | Linf(m)=5.018e-07 | checkpoint=33 | opt=7.792 s | original total=7.953 s | L2(1000)=9.796e-09 | Linf(1000)=3.570e-08 | extra 1000-refit=0.219 s
[m= 800 | seed  37/100] ... p*=[1.819696, 9.867239] | L2(m)=5.388e-09 | Linf(m)=2.176e-08 | checkpoint=13 | opt=7.320 s | original total=7.469 s | L2(1000)=2.746e-09 | Linf(1000)=8.680e-09 | extra 1000-refit=0.236 s
[m= 800 | seed  38/100] ... p*=[1.893809, 9.062554] | L2(m)=4.860e-09 | Linf(m)=1.844e-08 | checkpoint=49 | opt=7.760 s | original total=7.932 s | L2(1000)=9.737e-10 | Linf(1000)=2.448e-09 | extra 1000-refit=0.242 s
[m= 800 | seed  39/100] ... p*=[1.885098, 9.923914] | L2(m)=1.676e-08 | Linf(m)=4.232e-08 | checkpoint=50 | opt=7.389 s | original total=7.590 s | L2(1000)=3.103e-09 | Linf(1000)=7.653e-09 | extra 1000-refit=0.236 s
[m= 800 | seed  40/100] ... p*=[2.101586, 8.128519] | L2(m)=3.802e-09 | Linf(m)=1.776e-08 | checkpoint=34 | opt=7.187 s | original total=7.337 s | L2(1000)=1.514e-09 | Linf(1000)=5.318e-09 | extra 1000-refit=0.236 s
[m= 800 | seed  41/100] ... p*=[2.053788, 10.089214] | L2(m)=1.317e-08 | Linf(m)=7.246e-08 | checkpoint=27 | opt=7.773 s | original total=7.941 s | L2(1000)=3.474e-09 | Linf(1000)=1.551e-08 | extra 1000-refit=0.234 s
[m= 800 | seed  42/100] ... p*=[2.156419, 7.267140] | L2(m)=4.769e-09 | Linf(m)=2.453e-08 | checkpoint=23 | opt=7.410 s | original total=7.564 s | L2(1000)=5.902e-10 | Linf(1000)=2.471e-09 | extra 1000-refit=0.228 s
[m= 800 | seed  43/100] ... p*=[1.682671, 8.578422] | L2(m)=4.278e-09 | Linf(m)=1.297e-08 | checkpoint=7 | opt=7.722 s | original total=7.887 s | L2(1000)=1.360e-09 | Linf(1000)=4.971e-09 | extra 1000-refit=0.237 s
[m= 800 | seed  44/100] ... p*=[1.495523, 8.979170] | L2(m)=3.806e-09 | Linf(m)=1.486e-08 | checkpoint=40 | opt=7.838 s | original total=8.006 s | L2(1000)=1.447e-09 | Linf(1000)=4.860e-09 | extra 1000-refit=0.250 s
[m= 800 | seed  45/100] ... p*=[1.943018, 7.469202] | L2(m)=3.436e-09 | Linf(m)=1.954e-08 | checkpoint=45 | opt=7.381 s | original total=7.563 s | L2(1000)=1.122e-09 | Linf(1000)=4.496e-09 | extra 1000-refit=0.239 s
[m= 800 | seed  46/100] ... p*=[1.383967, 10.457224] | L2(m)=2.122e-08 | Linf(m)=4.268e-08 | checkpoint=3 | opt=7.624 s | original total=7.813 s | L2(1000)=3.743e-09 | Linf(1000)=9.886e-09 | extra 1000-refit=0.221 s
[m= 800 | seed  47/100] ... p*=[1.771908, 8.194087] | L2(m)=2.770e-09 | Linf(m)=1.385e-08 | checkpoint=49 | opt=7.249 s | original total=7.420 s | L2(1000)=8.516e-10 | Linf(1000)=2.353e-09 | extra 1000-refit=0.241 s
[m= 800 | seed  48/100] ... p*=[1.829420, 8.392243] | L2(m)=4.435e-09 | Linf(m)=2.001e-08 | checkpoint=6 | opt=7.476 s | original total=7.656 s | L2(1000)=2.492e-09 | Linf(1000)=4.859e-09 | extra 1000-refit=0.246 s
[m= 800 | seed  49/100] ... p*=[2.197458, 8.550531] | L2(m)=4.536e-09 | Linf(m)=2.787e-08 | checkpoint=46 | opt=7.329 s | original total=7.507 s | L2(1000)=2.249e-09 | Linf(1000)=8.180e-09 | extra 1000-refit=0.242 s
[m= 800 | seed  50/100] ... p*=[1.739241, 8.766885] | L2(m)=3.825e-09 | Linf(m)=2.185e-08 | checkpoint=34 | opt=7.926 s | original total=8.104 s | L2(1000)=2.296e-09 | Linf(1000)=6.303e-09 | extra 1000-refit=0.222 s
[m= 800 | seed  51/100] ... p*=[1.596004, 8.501955] | L2(m)=2.153e-09 | Linf(m)=8.200e-09 | checkpoint=50 | opt=7.441 s | original total=7.606 s | L2(1000)=5.724e-10 | Linf(1000)=1.222e-09 | extra 1000-refit=0.232 s
[m= 800 | seed  52/100] ... p*=[1.697685, 10.839402] | L2(m)=2.767e-08 | Linf(m)=8.076e-08 | checkpoint=7 | opt=7.663 s | original total=7.829 s | L2(1000)=1.305e-08 | Linf(1000)=3.148e-08 | extra 1000-refit=0.234 s
[m= 800 | seed  53/100] ... p*=[2.297910, 10.360515] | L2(m)=2.612e-08 | Linf(m)=1.817e-07 | checkpoint=43 | opt=7.286 s | original total=7.456 s | L2(1000)=3.520e-09 | Linf(1000)=9.906e-09 | extra 1000-refit=0.214 s
[m= 800 | seed  54/100] ... p*=[2.003203, 11.765916] | L2(m)=4.449e-08 | Linf(m)=1.546e-07 | checkpoint=15 | opt=7.504 s | original total=7.664 s | L2(1000)=7.219e-09 | Linf(1000)=2.728e-08 | extra 1000-refit=0.227 s
[m= 800 | seed  55/100] ... p*=[1.615923, 8.464198] | L2(m)=3.711e-09 | Linf(m)=1.531e-08 | checkpoint=3 | opt=7.861 s | original total=8.044 s | L2(1000)=1.341e-09 | Linf(1000)=5.311e-09 | extra 1000-refit=0.245 s
[m= 800 | seed  56/100] ... p*=[2.096316, 9.980511] | L2(m)=1.084e-08 | Linf(m)=6.510e-08 | checkpoint=8 | opt=8.040 s | original total=8.210 s | L2(1000)=3.387e-09 | Linf(1000)=1.804e-08 | extra 1000-refit=0.210 s
[m= 800 | seed  57/100] ... p*=[1.809970, 8.981041] | L2(m)=7.199e-09 | Linf(m)=2.694e-08 | checkpoint=50 | opt=7.400 s | original total=7.574 s | L2(1000)=1.288e-09 | Linf(1000)=3.411e-09 | extra 1000-refit=0.244 s
[m= 800 | seed  58/100] ... p*=[2.215757, 10.653595] | L2(m)=2.917e-08 | Linf(m)=1.170e-07 | checkpoint=46 | opt=7.932 s | original total=8.089 s | L2(1000)=2.817e-09 | Linf(1000)=1.041e-08 | extra 1000-refit=0.237 s
[m= 800 | seed  59/100] ... p*=[1.946630, 9.354576] | L2(m)=4.143e-09 | Linf(m)=2.117e-08 | checkpoint=46 | opt=7.587 s | original total=7.766 s | L2(1000)=2.103e-09 | Linf(1000)=6.478e-09 | extra 1000-refit=0.243 s
[m= 800 | seed  60/100] ... p*=[1.843156, 11.061508] | L2(m)=2.285e-08 | Linf(m)=6.706e-08 | checkpoint=26 | opt=7.808 s | original total=7.992 s | L2(1000)=3.395e-09 | Linf(1000)=1.178e-08 | extra 1000-refit=0.240 s
[m= 800 | seed  61/100] ... p*=[1.900318, 9.002256] | L2(m)=4.748e-09 | Linf(m)=2.594e-08 | checkpoint=43 | opt=7.390 s | original total=7.548 s | L2(1000)=7.061e-10 | Linf(1000)=2.647e-09 | extra 1000-refit=0.229 s
[m= 800 | seed  62/100] ... p*=[1.805099, 8.113062] | L2(m)=3.548e-09 | Linf(m)=2.251e-08 | checkpoint=8 | opt=7.376 s | original total=7.535 s | L2(1000)=7.787e-10 | Linf(1000)=2.877e-09 | extra 1000-refit=0.238 s
[m= 800 | seed  63/100] ... p*=[1.940801, 12.559922] | L2(m)=7.094e-08 | Linf(m)=3.337e-07 | checkpoint=33 | opt=7.740 s | original total=7.901 s | L2(1000)=6.388e-09 | Linf(1000)=2.702e-08 | extra 1000-refit=0.217 s
[m= 800 | seed  64/100] ... p*=[1.881875, 9.460505] | L2(m)=4.376e-09 | Linf(m)=1.472e-08 | checkpoint=31 | opt=7.309 s | original total=7.472 s | L2(1000)=1.775e-09 | Linf(1000)=4.961e-09 | extra 1000-refit=0.220 s
[m= 800 | seed  65/100] ... p*=[1.857986, 10.187221] | L2(m)=1.645e-08 | Linf(m)=7.646e-08 | checkpoint=29 | opt=7.891 s | original total=8.067 s | L2(1000)=1.847e-09 | Linf(1000)=7.356e-09 | extra 1000-refit=0.246 s
[m= 800 | seed  66/100] ... p*=[1.921038, 9.439480] | L2(m)=3.632e-09 | Linf(m)=1.453e-08 | checkpoint=33 | opt=7.804 s | original total=7.969 s | L2(1000)=1.441e-09 | Linf(1000)=4.316e-09 | extra 1000-refit=0.230 s
[m= 800 | seed  67/100] ... p*=[1.577851, 7.543989] | L2(m)=2.747e-09 | Linf(m)=2.247e-08 | checkpoint=9 | opt=7.230 s | original total=7.405 s | L2(1000)=1.192e-09 | Linf(1000)=5.339e-09 | extra 1000-refit=0.257 s
[m= 800 | seed  68/100] ... p*=[2.198818, 10.194861] | L2(m)=2.524e-08 | Linf(m)=1.343e-07 | checkpoint=39 | opt=7.884 s | original total=8.050 s | L2(1000)=1.185e-08 | Linf(1000)=2.524e-08 | extra 1000-refit=0.245 s
[m= 800 | seed  69/100] ... p*=[1.754509, 10.356050] | L2(m)=1.844e-08 | Linf(m)=6.532e-08 | checkpoint=6 | opt=7.273 s | original total=7.432 s | L2(1000)=6.861e-09 | Linf(1000)=2.684e-08 | extra 1000-refit=0.232 s
[m= 800 | seed  70/100] ... p*=[1.673785, 7.899053] | L2(m)=3.943e-09 | Linf(m)=2.790e-08 | checkpoint=50 | opt=7.999 s | original total=8.181 s | L2(1000)=1.250e-09 | Linf(1000)=6.100e-09 | extra 1000-refit=0.240 s
[m= 800 | seed  71/100] ... p*=[1.615411, 9.824975] | L2(m)=6.146e-09 | Linf(m)=2.649e-08 | checkpoint=48 | opt=7.447 s | original total=7.627 s | L2(1000)=2.660e-09 | Linf(1000)=6.717e-09 | extra 1000-refit=0.231 s
[m= 800 | seed  72/100] ... p*=[2.049611, 9.792372] | L2(m)=6.410e-09 | Linf(m)=2.422e-08 | checkpoint=2 | opt=7.711 s | original total=7.881 s | L2(1000)=1.157e-09 | Linf(1000)=2.555e-09 | extra 1000-refit=0.254 s
[m= 800 | seed  73/100] ... p*=[2.280555, 8.509448] | L2(m)=1.123e-08 | Linf(m)=7.241e-08 | checkpoint=2 | opt=7.302 s | original total=7.459 s | L2(1000)=1.436e-09 | Linf(1000)=6.712e-09 | extra 1000-refit=0.221 s
[m= 800 | seed  74/100] ... p*=[1.437894, 8.088595] | L2(m)=1.409e-09 | Linf(m)=5.936e-09 | checkpoint=3 | opt=8.127 s | original total=8.299 s | L2(1000)=1.128e-09 | Linf(1000)=3.250e-09 | extra 1000-refit=0.226 s
[m= 800 | seed  75/100] ... p*=[1.795290, 8.602986] | L2(m)=3.237e-09 | Linf(m)=1.331e-08 | checkpoint=20 | opt=7.516 s | original total=7.684 s | L2(1000)=7.166e-10 | Linf(1000)=3.192e-09 | extra 1000-refit=0.222 s
[m= 800 | seed  76/100] ... p*=[1.692607, 8.045590] | L2(m)=7.043e-09 | Linf(m)=2.827e-08 | checkpoint=35 | opt=8.036 s | original total=8.198 s | L2(1000)=1.265e-09 | Linf(1000)=3.941e-09 | extra 1000-refit=0.248 s
[m= 800 | seed  77/100] ... p*=[1.890095, 11.816876] | L2(m)=2.517e-08 | Linf(m)=6.955e-08 | checkpoint=29 | opt=7.738 s | original total=7.902 s | L2(1000)=9.151e-09 | Linf(1000)=2.546e-08 | extra 1000-refit=0.217 s
[m= 800 | seed  78/100] ... p*=[1.782176, 10.803635] | L2(m)=1.225e-08 | Linf(m)=6.496e-08 | checkpoint=7 | opt=7.843 s | original total=8.012 s | L2(1000)=1.630e-09 | Linf(1000)=8.642e-09 | extra 1000-refit=0.237 s
[m= 800 | seed  79/100] ... p*=[1.600657, 10.464627] | L2(m)=1.004e-08 | Linf(m)=4.495e-08 | checkpoint=3 | opt=7.413 s | original total=7.592 s | L2(1000)=3.275e-09 | Linf(1000)=7.676e-09 | extra 1000-refit=0.257 s
[m= 800 | seed  80/100] ... p*=[2.199408, 7.669821] | L2(m)=5.461e-09 | Linf(m)=2.400e-08 | checkpoint=48 | opt=7.971 s | original total=8.131 s | L2(1000)=7.439e-10 | Linf(1000)=2.719e-09 | extra 1000-refit=0.224 s
[m= 800 | seed  81/100] ... p*=[2.180131, 12.359482] | L2(m)=8.777e-08 | Linf(m)=6.453e-07 | checkpoint=19 | opt=7.266 s | original total=7.426 s | L2(1000)=1.776e-08 | Linf(1000)=6.324e-08 | extra 1000-refit=0.226 s
[m= 800 | seed  82/100] ... p*=[1.678680, 10.416455] | L2(m)=8.010e-09 | Linf(m)=4.233e-08 | checkpoint=3 | opt=7.838 s | original total=8.029 s | L2(1000)=4.833e-09 | Linf(1000)=1.263e-08 | extra 1000-refit=0.246 s
[m= 800 | seed  83/100] ... p*=[1.257255, 9.393555] | L2(m)=1.133e-08 | Linf(m)=2.946e-08 | checkpoint=45 | opt=7.362 s | original total=7.533 s | L2(1000)=3.349e-09 | Linf(1000)=1.153e-08 | extra 1000-refit=0.255 s
[m= 800 | seed  84/100] ... p*=[2.402225, 6.166673] | L2(m)=9.445e-09 | Linf(m)=4.013e-08 | checkpoint=2 | opt=7.222 s | original total=7.407 s | L2(1000)=2.255e-09 | Linf(1000)=8.277e-09 | extra 1000-refit=0.236 s
[m= 800 | seed  85/100] ... p*=[2.026958, 7.987641] | L2(m)=3.152e-09 | Linf(m)=1.760e-08 | checkpoint=27 | opt=7.978 s | original total=8.153 s | L2(1000)=1.026e-09 | Linf(1000)=3.821e-09 | extra 1000-refit=0.260 s
[m= 800 | seed  86/100] ... p*=[2.492154, 9.010787] | L2(m)=2.024e-08 | Linf(m)=9.142e-08 | checkpoint=38 | opt=7.437 s | original total=7.597 s | L2(1000)=2.137e-09 | Linf(1000)=8.250e-09 | extra 1000-refit=0.253 s
[m= 800 | seed  87/100] ... p*=[1.772601, 10.111782] | L2(m)=5.702e-09 | Linf(m)=1.572e-08 | checkpoint=3 | opt=7.718 s | original total=7.883 s | L2(1000)=1.412e-09 | Linf(1000)=5.155e-09 | extra 1000-refit=0.240 s
[m= 800 | seed  88/100] ... p*=[1.562726, 11.457461] | L2(m)=1.111e-08 | Linf(m)=3.905e-08 | checkpoint=50 | opt=7.889 s | original total=8.093 s | L2(1000)=1.777e-08 | Linf(1000)=4.815e-08 | extra 1000-refit=0.228 s
[m= 800 | seed  89/100] ... p*=[1.721775, 8.105697] | L2(m)=2.887e-09 | Linf(m)=1.465e-08 | checkpoint=38 | opt=7.355 s | original total=7.522 s | L2(1000)=1.464e-09 | Linf(1000)=4.911e-09 | extra 1000-refit=0.252 s
[m= 800 | seed  90/100] ... p*=[1.851556, 9.187788] | L2(m)=3.155e-09 | Linf(m)=1.290e-08 | checkpoint=47 | opt=7.846 s | original total=8.023 s | L2(1000)=1.972e-09 | Linf(1000)=5.722e-09 | extra 1000-refit=0.258 s
[m= 800 | seed  91/100] ... p*=[1.604834, 9.761368] | L2(m)=8.373e-09 | Linf(m)=2.430e-08 | checkpoint=3 | opt=7.244 s | original total=7.402 s | L2(1000)=1.253e-09 | Linf(1000)=4.006e-09 | extra 1000-refit=0.256 s
[m= 800 | seed  92/100] ... p*=[1.871904, 8.163990] | L2(m)=6.497e-09 | Linf(m)=1.866e-08 | checkpoint=45 | opt=7.761 s | original total=7.932 s | L2(1000)=8.620e-10 | Linf(1000)=4.543e-09 | extra 1000-refit=0.230 s
[m= 800 | seed  93/100] ... p*=[1.643923, 8.750956] | L2(m)=1.694e-09 | Linf(m)=7.718e-09 | checkpoint=47 | opt=7.401 s | original total=7.574 s | L2(1000)=1.325e-09 | Linf(1000)=3.695e-09 | extra 1000-refit=0.225 s
[m= 800 | seed  94/100] ... p*=[1.902230, 11.177930] | L2(m)=3.009e-08 | Linf(m)=1.786e-07 | checkpoint=50 | opt=7.982 s | original total=8.162 s | L2(1000)=4.424e-09 | Linf(1000)=1.117e-08 | extra 1000-refit=0.208 s
[m= 800 | seed  95/100] ... p*=[2.173388, 8.956432] | L2(m)=1.045e-08 | Linf(m)=6.069e-08 | checkpoint=31 | opt=7.401 s | original total=7.561 s | L2(1000)=1.469e-09 | Linf(1000)=4.516e-09 | extra 1000-refit=0.223 s
[m= 800 | seed  96/100] ... p*=[2.614575, 8.860502] | L2(m)=2.269e-08 | Linf(m)=1.752e-07 | checkpoint=9 | opt=8.028 s | original total=8.213 s | L2(1000)=2.312e-09 | Linf(1000)=8.009e-09 | extra 1000-refit=0.225 s
[m= 800 | seed  97/100] ... p*=[1.661396, 10.085438] | L2(m)=4.285e-09 | Linf(m)=1.254e-08 | checkpoint=3 | opt=7.588 s | original total=7.771 s | L2(1000)=1.799e-09 | Linf(1000)=5.096e-09 | extra 1000-refit=0.241 s
[m= 800 | seed  98/100] ... p*=[1.830975, 7.945520] | L2(m)=3.612e-09 | Linf(m)=3.409e-08 | checkpoint=48 | opt=8.005 s | original total=8.186 s | L2(1000)=1.066e-09 | Linf(1000)=4.581e-09 | extra 1000-refit=0.220 s
[m= 800 | seed  99/100] ... p*=[1.421773, 10.168819] | L2(m)=9.484e-09 | Linf(m)=2.956e-08 | checkpoint=43 | opt=7.910 s | original total=8.098 s | L2(1000)=4.917e-09 | Linf(1000)=2.021e-08 | extra 1000-refit=0.238 s
[m= 800 | seed 100/100] ... p*=[1.473489, 10.760057] | L2(m)=9.280e-09 | Linf(m)=4.242e-08 | checkpoint=4 | opt=7.764 s | original total=7.928 s | L2(1000)=6.078e-09 | Linf(1000)=1.567e-08 | extra 1000-refit=0.242 s

--------------------------------------------------------------------
Summary for training m = 800
--------------------------------------------------------------------
Original successful runs = 100 / 100
Expanded successful runs = 100 / 100

Original final refit: m = 800

Relative L2
Mean   = 1.284112e-08
Std    = 1.862125e-08
Median = 6.277607e-09
Best   = 1.408767e-09
Worst  = 1.415108e-07

Relative Linf
Mean   = 5.709638e-08
Std    = 9.025397e-08
Median = 2.621463e-08
Best   = 5.693004e-09
Worst  = 6.452525e-07

p1*
Mean = 1.827156
Std  = 0.277060

p2*
Mean = 9.372983
Std  = 1.280699

Mean selected checkpoint = 29.500

Original timing only
Mean basis time        = 0.000439 s
Mean PDE time          = 0.000240 s
Mean cache time        = 0.001537 s
Mean optimization time = 7.652045 s
Mean assembly time     = 0.051290 s
Mean final LS time     = 0.095730 s
Mean test time         = 0.023001 s
Mean original total    = 7.824368 s

Expanded final refit: training m = 800 -> final m = 1000

Expanded relative L2
Mean   = 3.158719e-09
Std    = 3.443299e-09
Median = 1.909651e-09
Best   = 4.116474e-10
Worst  = 1.776993e-08

Expanded relative Linf
Mean   = 9.906890e-09
Std    = 1.038815e-08
Median = 6.201394e-09
Best   = 1.221535e-09
Worst  = 6.323736e-08

Additional m -> 1000 timing only
Mean 1000-basis time    = 0.000503 s
Mean 1000-assembly time = 0.067091 s
Mean 1000-final LS time = 0.138842 s
Mean 1000-test time     = 0.030231 s
Mean extra refit total = 0.236689 s
--------------------------------------------------------------------

####################################################################
Feature number 5 / 5
Training m = 1000
Expanded final m = 1000
Initial p = [4.000000, 8.000000]
lambda = 1.000e-07
####################################################################

[m=1000 | seed   1/100] ... p*=[1.728914, 8.871208] | L2(m)=2.060e-09 | Linf(m)=5.036e-09 | checkpoint=46 | opt=11.901 s | original total=12.129 s | L2(1000)=2.060e-09 | Linf(1000)=5.036e-09 | extra 1000-refit=0.239 s
[m=1000 | seed   2/100] ... p*=[1.639345, 8.492952] | L2(m)=2.215e-09 | Linf(m)=7.900e-09 | checkpoint=34 | opt=12.197 s | original total=12.430 s | L2(1000)=2.215e-09 | Linf(1000)=7.900e-09 | extra 1000-refit=0.245 s
[m=1000 | seed   3/100] ... p*=[4.000000, 8.000000] | L2(m)=2.005e-07 | Linf(m)=7.484e-07 | checkpoint=0 | opt=12.300 s | original total=12.539 s | L2(1000)=2.005e-07 | Linf(1000)=7.484e-07 | extra 1000-refit=0.235 s
[m=1000 | seed   4/100] ... p*=[1.904448, 9.373950] | L2(m)=3.708e-09 | Linf(m)=1.049e-08 | checkpoint=23 | opt=12.210 s | original total=12.437 s | L2(1000)=3.708e-09 | Linf(1000)=1.049e-08 | extra 1000-refit=0.233 s
[m=1000 | seed   5/100] ... p*=[1.807399, 10.548664] | L2(m)=5.463e-09 | Linf(m)=9.646e-09 | checkpoint=31 | opt=12.020 s | original total=12.253 s | L2(1000)=5.463e-09 | Linf(1000)=9.646e-09 | extra 1000-refit=0.239 s
[m=1000 | seed   6/100] ... p*=[1.743393, 10.205590] | L2(m)=2.650e-09 | Linf(m)=6.538e-09 | checkpoint=3 | opt=12.240 s | original total=12.478 s | L2(1000)=2.650e-09 | Linf(1000)=6.538e-09 | extra 1000-refit=0.236 s
[m=1000 | seed   7/100] ... p*=[2.040689, 11.703792] | L2(m)=8.741e-09 | Linf(m)=2.443e-08 | checkpoint=32 | opt=12.186 s | original total=12.401 s | L2(1000)=8.741e-09 | Linf(1000)=2.443e-08 | extra 1000-refit=0.233 s
[m=1000 | seed   8/100] ... p*=[1.973053, 11.795152] | L2(m)=8.971e-09 | Linf(m)=2.049e-08 | checkpoint=8 | opt=12.080 s | original total=12.296 s | L2(1000)=8.971e-09 | Linf(1000)=2.049e-08 | extra 1000-refit=0.233 s
[m=1000 | seed   9/100] ... p*=[2.213045, 9.836272] | L2(m)=5.065e-09 | Linf(m)=1.584e-08 | checkpoint=2 | opt=12.194 s | original total=12.416 s | L2(1000)=5.065e-09 | Linf(1000)=1.584e-08 | extra 1000-refit=0.237 s
[m=1000 | seed  10/100] ... p*=[2.647188, 8.669475] | L2(m)=1.835e-09 | Linf(m)=6.223e-09 | checkpoint=35 | opt=12.184 s | original total=12.401 s | L2(1000)=1.835e-09 | Linf(1000)=6.223e-09 | extra 1000-refit=0.241 s
[m=1000 | seed  11/100] ... p*=[2.301281, 9.796401] | L2(m)=2.381e-09 | Linf(m)=7.916e-09 | checkpoint=23 | opt=12.185 s | original total=12.433 s | L2(1000)=2.381e-09 | Linf(1000)=7.916e-09 | extra 1000-refit=0.233 s
[m=1000 | seed  12/100] ... p*=[1.889946, 10.898461] | L2(m)=4.969e-09 | Linf(m)=1.112e-08 | checkpoint=29 | opt=12.204 s | original total=12.443 s | L2(1000)=4.969e-09 | Linf(1000)=1.112e-08 | extra 1000-refit=0.214 s
[m=1000 | seed  13/100] ... p*=[1.795895, 11.513966] | L2(m)=2.353e-09 | Linf(m)=1.053e-08 | checkpoint=7 | opt=11.582 s | original total=11.810 s | L2(1000)=2.353e-09 | Linf(1000)=1.053e-08 | extra 1000-refit=0.243 s
[m=1000 | seed  14/100] ... p*=[1.721935, 9.993515] | L2(m)=5.127e-09 | Linf(m)=1.262e-08 | checkpoint=48 | opt=12.275 s | original total=12.551 s | L2(1000)=5.127e-09 | Linf(1000)=1.262e-08 | extra 1000-refit=0.267 s
[m=1000 | seed  15/100] ... p*=[2.019896, 8.604046] | L2(m)=1.249e-09 | Linf(m)=3.427e-09 | checkpoint=34 | opt=12.912 s | original total=13.174 s | L2(1000)=1.249e-09 | Linf(1000)=3.427e-09 | extra 1000-refit=0.247 s
[m=1000 | seed  16/100] ... p*=[1.448480, 10.303114] | L2(m)=5.341e-09 | Linf(m)=1.149e-08 | checkpoint=3 | opt=12.782 s | original total=13.075 s | L2(1000)=5.341e-09 | Linf(1000)=1.149e-08 | extra 1000-refit=0.267 s
[m=1000 | seed  17/100] ... p*=[1.922394, 8.662930] | L2(m)=2.025e-09 | Linf(m)=5.844e-09 | checkpoint=36 | opt=12.805 s | original total=13.057 s | L2(1000)=2.025e-09 | Linf(1000)=5.844e-09 | extra 1000-refit=0.257 s
[m=1000 | seed  18/100] ... p*=[2.345799, 9.665559] | L2(m)=1.976e-09 | Linf(m)=5.622e-09 | checkpoint=43 | opt=13.189 s | original total=13.500 s | L2(1000)=1.976e-09 | Linf(1000)=5.622e-09 | extra 1000-refit=0.243 s
[m=1000 | seed  19/100] ... p*=[2.080632, 9.456147] | L2(m)=2.222e-09 | Linf(m)=6.979e-09 | checkpoint=34 | opt=13.017 s | original total=13.270 s | L2(1000)=2.222e-09 | Linf(1000)=6.979e-09 | extra 1000-refit=0.254 s
[m=1000 | seed  20/100] ... p*=[2.172955, 9.854535] | L2(m)=2.622e-09 | Linf(m)=9.345e-09 | checkpoint=49 | opt=12.646 s | original total=12.889 s | L2(1000)=2.622e-09 | Linf(1000)=9.345e-09 | extra 1000-refit=0.234 s
[m=1000 | seed  21/100] ... p*=[1.474636, 10.591786] | L2(m)=8.869e-09 | Linf(m)=1.647e-08 | checkpoint=3 | opt=12.875 s | original total=13.133 s | L2(1000)=8.869e-09 | Linf(1000)=1.647e-08 | extra 1000-refit=0.250 s
[m=1000 | seed  22/100] ... p*=[1.887428, 9.005026] | L2(m)=9.144e-10 | Linf(m)=3.330e-09 | checkpoint=43 | opt=13.011 s | original total=13.276 s | L2(1000)=9.144e-10 | Linf(1000)=3.330e-09 | extra 1000-refit=0.273 s
[m=1000 | seed  23/100] ... p*=[1.728813, 10.050575] | L2(m)=3.833e-09 | Linf(m)=1.213e-08 | checkpoint=6 | opt=13.036 s | original total=13.316 s | L2(1000)=3.833e-09 | Linf(1000)=1.213e-08 | extra 1000-refit=0.286 s
[m=1000 | seed  24/100] ... p*=[2.053631, 9.539812] | L2(m)=2.170e-09 | Linf(m)=4.889e-09 | checkpoint=43 | opt=13.008 s | original total=13.277 s | L2(1000)=2.170e-09 | Linf(1000)=4.889e-09 | extra 1000-refit=0.252 s
[m=1000 | seed  25/100] ... p*=[2.133436, 9.706934] | L2(m)=2.271e-09 | Linf(m)=6.663e-09 | checkpoint=2 | opt=13.214 s | original total=13.480 s | L2(1000)=2.271e-09 | Linf(1000)=6.663e-09 | extra 1000-refit=0.261 s
[m=1000 | seed  26/100] ... p*=[1.720048, 7.651671] | L2(m)=9.160e-10 | Linf(m)=5.606e-09 | checkpoint=11 | opt=13.247 s | original total=13.508 s | L2(1000)=9.160e-10 | Linf(1000)=5.606e-09 | extra 1000-refit=0.284 s
[m=1000 | seed  27/100] ... p*=[2.183699, 9.587648] | L2(m)=4.792e-09 | Linf(m)=1.541e-08 | checkpoint=2 | opt=12.858 s | original total=13.121 s | L2(1000)=4.792e-09 | Linf(1000)=1.541e-08 | extra 1000-refit=0.252 s
[m=1000 | seed  28/100] ... p*=[1.991360, 11.628087] | L2(m)=1.114e-08 | Linf(m)=2.479e-08 | checkpoint=35 | opt=12.743 s | original total=12.983 s | L2(1000)=1.114e-08 | Linf(1000)=2.479e-08 | extra 1000-refit=0.252 s
[m=1000 | seed  29/100] ... p*=[1.937196, 8.935473] | L2(m)=7.655e-10 | Linf(m)=2.290e-09 | checkpoint=32 | opt=12.796 s | original total=13.056 s | L2(1000)=7.655e-10 | Linf(1000)=2.290e-09 | extra 1000-refit=0.256 s
[m=1000 | seed  30/100] ... p*=[2.079002, 10.137526] | L2(m)=2.546e-09 | Linf(m)=7.950e-09 | checkpoint=39 | opt=12.563 s | original total=12.806 s | L2(1000)=2.546e-09 | Linf(1000)=7.950e-09 | extra 1000-refit=0.243 s
[m=1000 | seed  31/100] ... p*=[1.810842, 10.971701] | L2(m)=5.752e-09 | Linf(m)=1.417e-08 | checkpoint=45 | opt=12.573 s | original total=12.838 s | L2(1000)=5.752e-09 | Linf(1000)=1.417e-08 | extra 1000-refit=0.251 s
[m=1000 | seed  32/100] ... p*=[2.085529, 11.026715] | L2(m)=3.879e-09 | Linf(m)=1.329e-08 | checkpoint=50 | opt=12.694 s | original total=12.934 s | L2(1000)=3.879e-09 | Linf(1000)=1.329e-08 | extra 1000-refit=0.263 s
[m=1000 | seed  33/100] ... p*=[2.531133, 8.393823] | L2(m)=2.013e-09 | Linf(m)=6.094e-09 | checkpoint=25 | opt=12.955 s | original total=13.197 s | L2(1000)=2.013e-09 | Linf(1000)=6.094e-09 | extra 1000-refit=0.240 s
[m=1000 | seed  34/100] ... p*=[1.946113, 10.879738] | L2(m)=7.738e-09 | Linf(m)=2.381e-08 | checkpoint=41 | opt=12.593 s | original total=12.845 s | L2(1000)=7.738e-09 | Linf(1000)=2.381e-08 | extra 1000-refit=0.242 s
[m=1000 | seed  35/100] ... p*=[2.329634, 7.414112] | L2(m)=1.103e-09 | Linf(m)=4.526e-09 | checkpoint=43 | opt=12.387 s | original total=12.611 s | L2(1000)=1.103e-09 | Linf(1000)=4.526e-09 | extra 1000-refit=0.243 s
[m=1000 | seed  36/100] ... p*=[1.632975, 12.351613] | L2(m)=9.259e-09 | Linf(m)=2.604e-08 | checkpoint=21 | opt=12.593 s | original total=12.839 s | L2(1000)=9.259e-09 | Linf(1000)=2.604e-08 | extra 1000-refit=0.231 s
[m=1000 | seed  37/100] ... p*=[2.105646, 10.146254] | L2(m)=1.656e-09 | Linf(m)=6.196e-09 | checkpoint=16 | opt=13.031 s | original total=13.281 s | L2(1000)=1.656e-09 | Linf(1000)=6.196e-09 | extra 1000-refit=0.253 s
[m=1000 | seed  38/100] ... p*=[2.223228, 10.257725] | L2(m)=4.417e-09 | Linf(m)=1.989e-08 | checkpoint=49 | opt=12.496 s | original total=12.739 s | L2(1000)=4.417e-09 | Linf(1000)=1.989e-08 | extra 1000-refit=0.229 s
[m=1000 | seed  39/100] ... p*=[2.061668, 11.261192] | L2(m)=3.762e-09 | Linf(m)=8.648e-09 | checkpoint=30 | opt=12.789 s | original total=13.030 s | L2(1000)=3.762e-09 | Linf(1000)=8.648e-09 | extra 1000-refit=0.243 s
[m=1000 | seed  40/100] ... p*=[1.838978, 12.156737] | L2(m)=1.827e-08 | Linf(m)=5.585e-08 | checkpoint=9 | opt=12.685 s | original total=12.914 s | L2(1000)=1.827e-08 | Linf(1000)=5.585e-08 | extra 1000-refit=0.249 s
[m=1000 | seed  41/100] ... p*=[1.931285, 11.005465] | L2(m)=9.407e-09 | Linf(m)=2.977e-08 | checkpoint=32 | opt=12.410 s | original total=12.637 s | L2(1000)=9.407e-09 | Linf(1000)=2.977e-08 | extra 1000-refit=0.232 s
[m=1000 | seed  42/100] ... p*=[1.889609, 9.968542] | L2(m)=1.124e-09 | Linf(m)=4.318e-09 | checkpoint=44 | opt=12.203 s | original total=12.434 s | L2(1000)=1.124e-09 | Linf(1000)=4.318e-09 | extra 1000-refit=0.218 s
[m=1000 | seed  43/100] ... p*=[1.737925, 7.779766] | L2(m)=5.854e-10 | Linf(m)=3.155e-09 | checkpoint=6 | opt=12.055 s | original total=12.308 s | L2(1000)=5.854e-10 | Linf(1000)=3.155e-09 | extra 1000-refit=0.240 s
[m=1000 | seed  44/100] ... p*=[2.055129, 10.503260] | L2(m)=1.471e-09 | Linf(m)=4.657e-09 | checkpoint=50 | opt=12.389 s | original total=12.630 s | L2(1000)=1.471e-09 | Linf(1000)=4.657e-09 | extra 1000-refit=0.235 s
[m=1000 | seed  45/100] ... p*=[1.837395, 8.825100] | L2(m)=1.652e-09 | Linf(m)=6.487e-09 | checkpoint=19 | opt=12.303 s | original total=12.551 s | L2(1000)=1.652e-09 | Linf(1000)=6.487e-09 | extra 1000-refit=0.256 s
[m=1000 | seed  46/100] ... p*=[1.432892, 10.546578] | L2(m)=4.899e-09 | Linf(m)=1.793e-08 | checkpoint=3 | opt=12.210 s | original total=12.440 s | L2(1000)=4.899e-09 | Linf(1000)=1.793e-08 | extra 1000-refit=0.236 s
[m=1000 | seed  47/100] ... p*=[3.038441, 9.575529] | L2(m)=3.919e-09 | Linf(m)=1.623e-08 | checkpoint=44 | opt=12.220 s | original total=12.437 s | L2(1000)=3.919e-09 | Linf(1000)=1.623e-08 | extra 1000-refit=0.241 s
[m=1000 | seed  48/100] ... p*=[1.936962, 10.957302] | L2(m)=6.321e-09 | Linf(m)=2.575e-08 | checkpoint=4 | opt=12.127 s | original total=12.347 s | L2(1000)=6.321e-09 | Linf(1000)=2.575e-08 | extra 1000-refit=0.230 s
[m=1000 | seed  49/100] ... p*=[1.881405, 9.946432] | L2(m)=2.652e-09 | Linf(m)=8.017e-09 | checkpoint=20 | opt=12.424 s | original total=12.685 s | L2(1000)=2.652e-09 | Linf(1000)=8.017e-09 | extra 1000-refit=0.214 s
[m=1000 | seed  50/100] ... p*=[1.979131, 8.529352] | L2(m)=1.567e-09 | Linf(m)=4.261e-09 | checkpoint=45 | opt=12.076 s | original total=12.309 s | L2(1000)=1.567e-09 | Linf(1000)=4.261e-09 | extra 1000-refit=0.222 s
[m=1000 | seed  51/100] ... p*=[2.266110, 9.110201] | L2(m)=1.325e-09 | Linf(m)=4.706e-09 | checkpoint=50 | opt=13.831 s | original total=14.114 s | L2(1000)=1.325e-09 | Linf(1000)=4.706e-09 | extra 1000-refit=0.218 s
[m=1000 | seed  52/100] ... p*=[2.210415, 9.882794] | L2(m)=5.204e-09 | Linf(m)=1.128e-08 | checkpoint=15 | opt=12.249 s | original total=12.475 s | L2(1000)=5.204e-09 | Linf(1000)=1.128e-08 | extra 1000-refit=0.237 s
[m=1000 | seed  53/100] ... p*=[2.209935, 10.212476] | L2(m)=2.900e-09 | Linf(m)=8.266e-09 | checkpoint=40 | opt=11.967 s | original total=12.188 s | L2(1000)=2.900e-09 | Linf(1000)=8.266e-09 | extra 1000-refit=0.224 s
[m=1000 | seed  54/100] ... p*=[2.099284, 9.232254] | L2(m)=9.358e-10 | Linf(m)=2.962e-09 | checkpoint=36 | opt=12.224 s | original total=12.480 s | L2(1000)=9.358e-10 | Linf(1000)=2.962e-09 | extra 1000-refit=0.223 s
[m=1000 | seed  55/100] ... p*=[2.093927, 9.471695] | L2(m)=2.451e-09 | Linf(m)=5.698e-09 | checkpoint=43 | opt=12.221 s | original total=12.459 s | L2(1000)=2.451e-09 | Linf(1000)=5.698e-09 | extra 1000-refit=0.235 s
[m=1000 | seed  56/100] ... p*=[1.799792, 7.967741] | L2(m)=1.347e-09 | Linf(m)=4.119e-09 | checkpoint=47 | opt=12.075 s | original total=12.312 s | L2(1000)=1.347e-09 | Linf(1000)=4.119e-09 | extra 1000-refit=0.244 s
[m=1000 | seed  57/100] ... p*=[2.138499, 7.663376] | L2(m)=1.388e-09 | Linf(m)=3.420e-09 | checkpoint=7 | opt=11.974 s | original total=12.242 s | L2(1000)=1.388e-09 | Linf(1000)=3.420e-09 | extra 1000-refit=0.229 s
[m=1000 | seed  58/100] ... p*=[2.167523, 9.561854] | L2(m)=4.993e-09 | Linf(m)=1.577e-08 | checkpoint=49 | opt=12.233 s | original total=12.463 s | L2(1000)=4.993e-09 | Linf(1000)=1.577e-08 | extra 1000-refit=0.229 s
[m=1000 | seed  59/100] ... p*=[2.034302, 10.521568] | L2(m)=4.061e-09 | Linf(m)=1.413e-08 | checkpoint=34 | opt=12.233 s | original total=12.473 s | L2(1000)=4.061e-09 | Linf(1000)=1.413e-08 | extra 1000-refit=0.232 s
[m=1000 | seed  60/100] ... p*=[2.209210, 9.753304] | L2(m)=2.816e-09 | Linf(m)=8.689e-09 | checkpoint=2 | opt=12.243 s | original total=12.502 s | L2(1000)=2.816e-09 | Linf(1000)=8.689e-09 | extra 1000-refit=0.232 s
[m=1000 | seed  61/100] ... p*=[1.809910, 10.715371] | L2(m)=3.364e-09 | Linf(m)=8.451e-09 | checkpoint=48 | opt=12.348 s | original total=12.572 s | L2(1000)=3.364e-09 | Linf(1000)=8.451e-09 | extra 1000-refit=0.235 s
[m=1000 | seed  62/100] ... p*=[2.406349, 11.768623] | L2(m)=8.156e-09 | Linf(m)=2.901e-08 | checkpoint=36 | opt=12.234 s | original total=12.457 s | L2(1000)=8.156e-09 | Linf(1000)=2.901e-08 | extra 1000-refit=0.226 s
[m=1000 | seed  63/100] ... p*=[1.761451, 12.639956] | L2(m)=6.374e-09 | Linf(m)=2.000e-08 | checkpoint=41 | opt=12.053 s | original total=12.276 s | L2(1000)=6.374e-09 | Linf(1000)=2.000e-08 | extra 1000-refit=0.231 s
[m=1000 | seed  64/100] ... p*=[2.082970, 11.528631] | L2(m)=4.312e-09 | Linf(m)=1.250e-08 | checkpoint=44 | opt=12.000 s | original total=12.233 s | L2(1000)=4.312e-09 | Linf(1000)=1.250e-08 | extra 1000-refit=0.249 s
[m=1000 | seed  65/100] ... p*=[1.664896, 10.385389] | L2(m)=3.753e-09 | Linf(m)=1.138e-08 | checkpoint=4 | opt=12.493 s | original total=12.733 s | L2(1000)=3.753e-09 | Linf(1000)=1.138e-08 | extra 1000-refit=0.227 s
[m=1000 | seed  66/100] ... p*=[1.678839, 10.558423] | L2(m)=3.486e-09 | Linf(m)=1.457e-08 | checkpoint=29 | opt=12.271 s | original total=12.503 s | L2(1000)=3.486e-09 | Linf(1000)=1.457e-08 | extra 1000-refit=0.216 s
[m=1000 | seed  67/100] ... p*=[1.997406, 9.880659] | L2(m)=2.388e-09 | Linf(m)=1.366e-08 | checkpoint=39 | opt=12.253 s | original total=12.461 s | L2(1000)=2.388e-09 | Linf(1000)=1.366e-08 | extra 1000-refit=0.207 s
[m=1000 | seed  68/100] ... p*=[2.131532, 8.658566] | L2(m)=1.996e-09 | Linf(m)=5.329e-09 | checkpoint=46 | opt=12.274 s | original total=12.500 s | L2(1000)=1.996e-09 | Linf(1000)=5.329e-09 | extra 1000-refit=0.233 s
[m=1000 | seed  69/100] ... p*=[2.035972, 7.420149] | L2(m)=5.962e-10 | Linf(m)=1.948e-09 | checkpoint=2 | opt=11.793 s | original total=12.035 s | L2(1000)=5.962e-10 | Linf(1000)=1.948e-09 | extra 1000-refit=0.256 s
[m=1000 | seed  70/100] ... p*=[1.684447, 9.368239] | L2(m)=4.964e-09 | Linf(m)=1.233e-08 | checkpoint=45 | opt=12.041 s | original total=12.267 s | L2(1000)=4.964e-09 | Linf(1000)=1.233e-08 | extra 1000-refit=0.242 s
[m=1000 | seed  71/100] ... p*=[1.826980, 10.129273] | L2(m)=2.306e-09 | Linf(m)=7.013e-09 | checkpoint=6 | opt=12.134 s | original total=12.361 s | L2(1000)=2.306e-09 | Linf(1000)=7.013e-09 | extra 1000-refit=0.220 s
[m=1000 | seed  72/100] ... p*=[1.693287, 11.281487] | L2(m)=3.619e-09 | Linf(m)=1.012e-08 | checkpoint=30 | opt=11.804 s | original total=12.042 s | L2(1000)=3.619e-09 | Linf(1000)=1.012e-08 | extra 1000-refit=0.220 s
[m=1000 | seed  73/100] ... p*=[2.031989, 11.658888] | L2(m)=1.604e-08 | Linf(m)=3.138e-08 | checkpoint=28 | opt=12.085 s | original total=12.298 s | L2(1000)=1.604e-08 | Linf(1000)=3.138e-08 | extra 1000-refit=0.214 s
[m=1000 | seed  74/100] ... p*=[2.919220, 10.211379] | L2(m)=9.113e-09 | Linf(m)=2.722e-08 | checkpoint=26 | opt=11.990 s | original total=12.227 s | L2(1000)=9.113e-09 | Linf(1000)=2.722e-08 | extra 1000-refit=0.240 s
[m=1000 | seed  75/100] ... p*=[1.510754, 11.511920] | L2(m)=5.688e-09 | Linf(m)=2.186e-08 | checkpoint=7 | opt=11.934 s | original total=12.158 s | L2(1000)=5.688e-09 | Linf(1000)=2.186e-08 | extra 1000-refit=0.215 s
[m=1000 | seed  76/100] ... p*=[2.107120, 9.576528] | L2(m)=1.825e-09 | Linf(m)=6.394e-09 | checkpoint=47 | opt=11.879 s | original total=12.103 s | L2(1000)=1.825e-09 | Linf(1000)=6.394e-09 | extra 1000-refit=0.248 s
[m=1000 | seed  77/100] ... p*=[2.213988, 11.718855] | L2(m)=8.198e-09 | Linf(m)=3.209e-08 | checkpoint=49 | opt=12.219 s | original total=12.441 s | L2(1000)=8.198e-09 | Linf(1000)=3.209e-08 | extra 1000-refit=0.226 s
[m=1000 | seed  78/100] ... p*=[1.508648, 10.499752] | L2(m)=2.666e-09 | Linf(m)=7.020e-09 | checkpoint=3 | opt=11.904 s | original total=12.174 s | L2(1000)=2.666e-09 | Linf(1000)=7.020e-09 | extra 1000-refit=0.253 s
[m=1000 | seed  79/100] ... p*=[1.909553, 8.423322] | L2(m)=6.925e-10 | Linf(m)=2.251e-09 | checkpoint=50 | opt=12.260 s | original total=12.504 s | L2(1000)=6.925e-10 | Linf(1000)=2.251e-09 | extra 1000-refit=0.227 s
[m=1000 | seed  80/100] ... p*=[1.553064, 8.183257] | L2(m)=9.507e-10 | Linf(m)=3.081e-09 | checkpoint=3 | opt=12.090 s | original total=12.346 s | L2(1000)=9.507e-10 | Linf(1000)=3.081e-09 | extra 1000-refit=0.236 s
[m=1000 | seed  81/100] ... p*=[2.455054, 8.777782] | L2(m)=2.993e-09 | Linf(m)=9.826e-09 | checkpoint=48 | opt=12.152 s | original total=12.402 s | L2(1000)=2.993e-09 | Linf(1000)=9.826e-09 | extra 1000-refit=0.232 s
[m=1000 | seed  82/100] ... p*=[1.752418, 11.088183] | L2(m)=7.743e-09 | Linf(m)=1.956e-08 | checkpoint=50 | opt=12.228 s | original total=12.458 s | L2(1000)=7.743e-09 | Linf(1000)=1.956e-08 | extra 1000-refit=0.237 s
[m=1000 | seed  83/100] ... p*=[2.106878, 10.015534] | L2(m)=2.529e-09 | Linf(m)=9.806e-09 | checkpoint=27 | opt=11.902 s | original total=12.162 s | L2(1000)=2.529e-09 | Linf(1000)=9.806e-09 | extra 1000-refit=0.250 s
[m=1000 | seed  84/100] ... p*=[2.217438, 9.641032] | L2(m)=2.157e-09 | Linf(m)=6.397e-09 | checkpoint=2 | opt=12.179 s | original total=12.424 s | L2(1000)=2.157e-09 | Linf(1000)=6.397e-09 | extra 1000-refit=0.213 s
[m=1000 | seed  85/100] ... p*=[1.875021, 11.587448] | L2(m)=6.877e-09 | Linf(m)=2.229e-08 | checkpoint=7 | opt=12.088 s | original total=12.316 s | L2(1000)=6.877e-09 | Linf(1000)=2.229e-08 | extra 1000-refit=0.275 s
[m=1000 | seed  86/100] ... p*=[1.751417, 9.373087] | L2(m)=1.454e-09 | Linf(m)=3.031e-09 | checkpoint=6 | opt=11.816 s | original total=12.055 s | L2(1000)=1.454e-09 | Linf(1000)=3.031e-09 | extra 1000-refit=0.268 s
[m=1000 | seed  87/100] ... p*=[2.199861, 9.171341] | L2(m)=1.748e-09 | Linf(m)=5.676e-09 | checkpoint=28 | opt=11.882 s | original total=12.120 s | L2(1000)=1.748e-09 | Linf(1000)=5.676e-09 | extra 1000-refit=0.233 s
[m=1000 | seed  88/100] ... p*=[2.234594, 10.594321] | L2(m)=5.042e-09 | Linf(m)=1.361e-08 | checkpoint=29 | opt=12.110 s | original total=12.332 s | L2(1000)=5.042e-09 | Linf(1000)=1.361e-08 | extra 1000-refit=0.223 s
[m=1000 | seed  89/100] ... p*=[2.217829, 8.346604] | L2(m)=6.933e-10 | Linf(m)=2.400e-09 | checkpoint=8 | opt=12.246 s | original total=12.473 s | L2(1000)=6.933e-10 | Linf(1000)=2.400e-09 | extra 1000-refit=0.223 s
[m=1000 | seed  90/100] ... p*=[2.061258, 9.476815] | L2(m)=2.041e-09 | Linf(m)=5.892e-09 | checkpoint=8 | opt=12.132 s | original total=12.377 s | L2(1000)=2.041e-09 | Linf(1000)=5.892e-09 | extra 1000-refit=0.228 s
[m=1000 | seed  91/100] ... p*=[1.771002, 8.449544] | L2(m)=7.227e-10 | Linf(m)=2.957e-09 | checkpoint=40 | opt=11.990 s | original total=12.209 s | L2(1000)=7.227e-10 | Linf(1000)=2.957e-09 | extra 1000-refit=0.242 s
[m=1000 | seed  92/100] ... p*=[1.853131, 12.535921] | L2(m)=2.145e-08 | Linf(m)=5.363e-08 | checkpoint=20 | opt=12.190 s | original total=12.407 s | L2(1000)=2.145e-08 | Linf(1000)=5.363e-08 | extra 1000-refit=0.209 s
[m=1000 | seed  93/100] ... p*=[1.828862, 10.565319] | L2(m)=3.621e-09 | Linf(m)=1.474e-08 | checkpoint=40 | opt=12.185 s | original total=12.420 s | L2(1000)=3.621e-09 | Linf(1000)=1.474e-08 | extra 1000-refit=0.225 s
[m=1000 | seed  94/100] ... p*=[1.868394, 8.556189] | L2(m)=1.310e-09 | Linf(m)=4.746e-09 | checkpoint=28 | opt=12.235 s | original total=12.489 s | L2(1000)=1.310e-09 | Linf(1000)=4.746e-09 | extra 1000-refit=0.239 s
[m=1000 | seed  95/100] ... p*=[1.966720, 7.770788] | L2(m)=1.514e-09 | Linf(m)=3.870e-09 | checkpoint=24 | opt=12.129 s | original total=12.379 s | L2(1000)=1.514e-09 | Linf(1000)=3.870e-09 | extra 1000-refit=0.238 s
[m=1000 | seed  96/100] ... p*=[2.008084, 11.150611] | L2(m)=6.635e-09 | Linf(m)=1.955e-08 | checkpoint=38 | opt=12.126 s | original total=12.361 s | L2(1000)=6.635e-09 | Linf(1000)=1.955e-08 | extra 1000-refit=0.220 s
[m=1000 | seed  97/100] ... p*=[1.458292, 8.742806] | L2(m)=1.797e-09 | Linf(m)=6.117e-09 | checkpoint=45 | opt=12.267 s | original total=12.519 s | L2(1000)=1.797e-09 | Linf(1000)=6.117e-09 | extra 1000-refit=0.233 s
[m=1000 | seed  98/100] ... p*=[1.833725, 10.807424] | L2(m)=7.917e-09 | Linf(m)=2.301e-08 | checkpoint=43 | opt=12.388 s | original total=12.639 s | L2(1000)=7.917e-09 | Linf(1000)=2.301e-08 | extra 1000-refit=0.217 s
[m=1000 | seed  99/100] ... p*=[1.793962, 10.561049] | L2(m)=4.172e-09 | Linf(m)=1.185e-08 | checkpoint=6 | opt=12.095 s | original total=12.320 s | L2(1000)=4.172e-09 | Linf(1000)=1.185e-08 | extra 1000-refit=0.215 s
[m=1000 | seed 100/100] ... p*=[1.667762, 9.899324] | L2(m)=1.316e-09 | Linf(m)=5.080e-09 | checkpoint=24 | opt=12.321 s | original total=12.552 s | L2(1000)=1.316e-09 | Linf(1000)=5.080e-09 | extra 1000-refit=0.236 s

--------------------------------------------------------------------
Summary for training m = 1000
--------------------------------------------------------------------
Original successful runs = 100 / 100
Expanded successful runs = 100 / 100

Original final refit: m = 1000

Relative L2
Mean   = 6.027601e-09
Std    = 1.996625e-08
Median = 2.740805e-09
Best   = 5.854425e-10
Worst  = 2.004872e-07

Relative Linf
Mean   = 1.939180e-08
Std    = 7.427207e-08
Median = 9.017037e-09
Best   = 1.948228e-09
Worst  = 7.483951e-07

p1*
Mean = 1.995509
Std  = 0.347296

p2*
Mean = 9.921843
Std  = 1.226929

Mean selected checkpoint = 27.590

Original timing only
Mean basis time        = 0.000447 s
Mean PDE time          = 0.000252 s
Mean cache time        = 0.001860 s
Mean optimization time = 12.336427 s
Mean assembly time     = 0.068896 s
Mean final LS time     = 0.139514 s
Mean test time         = 0.030098 s
Mean original total    = 12.577582 s

Expanded final refit: training m = 1000 -> final m = 1000

Expanded relative L2
Mean   = 6.027601e-09
Std    = 1.996625e-08
Median = 2.740805e-09
Best   = 5.854425e-10
Worst  = 2.004872e-07

Expanded relative Linf
Mean   = 1.939180e-08
Std    = 7.427207e-08
Median = 9.017037e-09
Best   = 1.948228e-09
Worst  = 7.483951e-07

Additional m -> 1000 timing only
Mean 1000-basis time    = 0.000490 s
Mean 1000-assembly time = 0.068576 s
Mean 1000-final LS time = 0.138355 s
Mean 1000-test time     = 0.030417 s
Mean extra refit total = 0.237861 s
--------------------------------------------------------------------


====================================================================
                         FINAL SUMMARY
====================================================================

    TrainingFeatures    ExpandedFeatures    num_success    num_expand_success     L2_mean        L2_std      L2_median      L2_best       L2_worst     L2_expand_mean    L2_expand_std    L2_expand_median    L2_expand_best    L2_expand_worst    Linf_mean      Linf_std     Linf_expand_mean    Linf_expand_std    p1_mean    p1_std     p2_mean    p2_std    selected_checkpoint_mean    selected_checkpoint_std    setup_mean    basis_mean     pde_mean     cache_mean    optimization_mean    final_refit_mean    assembly_mean    final_ls_mean    test_mean    total_mean    total_std    expand_basis_mean    expand_assembly_mean    expand_final_ls_mean    expand_test_mean    expand_total_mean    expand_total_std
    ________________    ________________    ___________    __________________    __________    __________    __________    __________    __________    ______________    _____________    ________________    ______________    _______________    __________    __________    ________________    _______________    _______    _______    _______    ______    ________________________    _______________________    __________    __________    __________    __________    _________________    ________________    _____________    _____________    _________    __________    _________    _________________    ____________________    ____________________    ________________    _________________    ________________

           200                1000              100               100             0.0025923     0.0029163     0.0015833    0.00018229      0.019561      1.8698e-09       1.5802e-09         1.4465e-09          5.049e-10        1.1925e-08        0.0077636     0.0079063       8.7778e-09         8.1653e-09       1.1072     0.23481    5.7397     1.4754              29.1                      19.925             0.0012694     0.00041852    0.00035685    0.00049401         1.1501              0.025481           0.01104         0.014441       0.0034282      1.1804      0.047696        0.00049906              0.073603                0.15551               0.029995             0.25968             0.016019    
           400                1000              100               100            1.1031e-05    2.0657e-05    1.6404e-06    1.8369e-07    0.00012794      2.2103e-09       3.5389e-09         9.6425e-10         4.2528e-10        2.0211e-08        4.756e-05    8.9856e-05       8.1777e-09         9.6654e-09       1.4174     0.25541    6.4924     1.8628             11.41                      15.004             0.0014287     0.00042327    0.00024759    0.00075788          2.788              0.060726           0.02271         0.038016       0.0065986      2.8569       0.12132        0.00050922              0.069908                 0.1519               0.029839             0.25218             0.016636    
           600                1000              100               100            1.8947e-07    3.3497e-07    8.3811e-08    6.4489e-09    2.6148e-06      1.6861e-09       1.9893e-09         1.0834e-09          3.563e-10        1.4847e-08       1.0594e-06     1.925e-06       5.5713e-09          4.698e-09        1.693     0.31826    7.9631     1.4597             26.81                      18.253             0.0017327     0.00043948     0.0002438     0.0010494          4.863               0.10796          0.038976         0.068986        0.011258       4.984       0.23054         0.0005183              0.070311                0.15008               0.029911             0.25084             0.016454    
           800                1000              100               100            1.2841e-08    1.8621e-08    6.2776e-09    1.4088e-09    1.4151e-07      3.1587e-09       3.4433e-09         1.9097e-09         4.1165e-10         1.777e-08       5.7096e-08    9.0254e-08       9.9069e-09         1.0388e-08       1.8272     0.27706     9.373     1.2807              29.5                      17.432             0.0022156     0.00043879    0.00023958     0.0015372          7.652               0.14702           0.05129          0.09573        0.023001      7.8244       0.27998        0.00050313              0.067091                0.13884               0.030231             0.23669             0.011807    
          1000                1000              100               100            6.0276e-09    1.9966e-08    2.7408e-09    5.8544e-10    2.0049e-07      6.0276e-09       1.9966e-08         2.7408e-09         5.8544e-10        2.0049e-07       1.9392e-08    7.4272e-08       1.9392e-08         7.4272e-08       1.9955      0.3473    9.9218     1.2269             27.59                      16.838             0.0025601     0.00044735    0.00025228     0.0018605         12.336               0.20841          0.068896          0.13951        0.030098      12.578       0.39396        0.00048968              0.068576                0.13836               0.030417             0.23786             0.015918    


=========================================================================================================================
 Train m    Mean L2(m)      Std L2(m)      Mean L2(m->1000)    Std L2(m->1000)    p1*      p2*
=========================================================================================================================
    200    2.592e-03       2.916e-03         1.870e-09           1.580e-09          1.107    5.740
    400    1.103e-05       2.066e-05         2.210e-09           3.539e-09          1.417    6.492
    600    1.895e-07       3.350e-07         1.686e-09           1.989e-09          1.693    7.963
    800    1.284e-08       1.862e-08         3.159e-09           3.443e-09          1.827    9.373
   1000    6.028e-09       1.997e-08         6.028e-09           1.997e-08          1.996    9.922
=========================================================================================================================

============================================================================================================
 Train m    Setup/s     Optim/s    FinalRefit(m)/s    Test(m)/s    OriginalTotal/s    Checkpoint
============================================================================================================
    200      0.0013      1.1501          0.0255          0.0034          1.1804          29.10
    400      0.0014      2.7880          0.0607          0.0066          2.8569          11.41
    600      0.0017      4.8630          0.1080          0.0113          4.9840          26.81
    800      0.0022      7.6520          0.1470          0.0230          7.8244          29.50
   1000      0.0026     12.3364          0.2084          0.0301         12.5776          27.59
============================================================================================================

==============================================================================================================
 Train m    Basis1000/s    Assembly1000/s    LS1000/s    Test1000/s    ExtraRefitTotal/s
==============================================================================================================
    200        0.0005            0.0736         0.1555        0.0300           0.2597
    400        0.0005            0.0699         0.1519        0.0298           0.2522
    600        0.0005            0.0703         0.1501        0.0299           0.2508
    800        0.0005            0.0671         0.1388        0.0302           0.2367
   1000        0.0005            0.0686         0.1384        0.0304           0.2379
==============================================================================================================

============================================================================================================================
 Train m    Basis       PDE       Cache      Optimization    Assembly(m)    FinalLS(m)    Test(m)     Total(m)
============================================================================================================================
    200     0.0004     0.0004     0.0005        1.1501         0.0110        0.0144      0.0034      1.1804
    400     0.0004     0.0002     0.0008        2.7880         0.0227        0.0380      0.0066      2.8569
    600     0.0004     0.0002     0.0010        4.8630         0.0390        0.0690      0.0113      4.9840
    800     0.0004     0.0002     0.0015        7.6520         0.0513        0.0957      0.0230      7.8244
   1000     0.0004     0.0003     0.0019       12.3364         0.0689        0.1395      0.0301     12.5776
============================================================================================================================

Results saved to:
/data/yangyou/matlab_code/AD-RaNN/ad-rann-complete-general/examples/poisson_2d/feature_number_study_results/feature_number_study_final.mat
/data/yangyou/matlab_code/AD-RaNN/ad-rann-complete-general/examples/poisson_2d/feature_number_study_results/feature_number_summary.csv
警告: 已忽略了负数 
> 位置：matlab.graphics.shape.internal.AxesLayoutManager>calculateLooseInset
位置: matlab.graphics.shape.internal/AxesLayoutManager/updateStartingLayoutPosition
位置: matlab.graphics.shape.internal/AxesLayoutManager/doUpdate 
