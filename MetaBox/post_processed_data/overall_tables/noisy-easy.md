The overall comparison on optimization performance among all baselines in our Repo. For each problem in **Noisy Synthetic easy** suites, we test each algorithms for $51$ independent runs and average the test results for presentation. ''Obj'' (smaller is better) indicates the final global best value. ''Gap'' (smaller is better) is the optimization gap away from the **SOTA** optimizer, which is CMA-ES algorithm in this paper. ''FEs'' indicates how many function evaluations an optimizer takes to get the ''Obj''. Under the consideration of runtime, we assign BayesianOptimizer 100 FEs. For RNN-OI, we also set the maxFEs to 100 respecting the original paper. Therefore, their ''FEs'' and ''Gap'' are not comparable.
<body>
    <table style="width:1200pt"> <!--StartFragment--> 
 <colgroup>
  <col width="64" span="25" style="width:48pt"> 
 </colgroup>
 <tbody>
  <tr height="19"> 
   <td class="xl63">Problem</td> 
   <td colspan="3" class="xl63">Sphere_moderate_gauss</td> 
   <td colspan="3" class="xl63">Rosenbrock_moderate_uniform</td> 
   <td colspan="3" class="xl63">Step_Ellipsoidal_cauchy</td> 
   <td colspan="3" class="xl63">Ellipsoidal_gauss</td> 
   <td colspan="3" class="xl63">Ellipsoidal_uniform</td> 
   <td colspan="3" class="xl63">Different_Powers_gauss</td> 
   <td colspan="3" class="xl63">Different_Powers_uniform</td> 
   <td colspan="3" class="xl63">Composite_Grie_rosen_gauss</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">metric</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
   <td class="xl63">Obj</td> 
   <td class="xl63">Gap</td> 
   <td class="xl63">FEs</td> 
  </tr> 
  <tr height="19"> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
   <td class></td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">DE_DDQN</td> 
   <td class>7.907e-9<br>(1.523e-9)</td> 
   <td class>0.000</td> 
   <td class>1.692e+4<br>(5.373e+2)</td> 
   <td class>2.913e+0<br>(2.859e+0)</td> 
   <td class>0.001</td> 
   <td class>1.977e+4<br>(7.700e+2)</td> 
   <td class>1.022e+2<br>(1.726e+2)</td> 
   <td class>0.153</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>3.243e+0<br>(3.705e+0)</td> 
   <td class>-2.944</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.522e+0<br>(5.679e+0)</td> 
   <td class>0.235</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.643e-2<br>(4.280e-2)</td> 
   <td class>-0.443</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.847e-2<br>(1.702e-2)</td> 
   <td class>1.270</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.896e-2<br>(8.325e-3)</td> 
   <td class>0.328</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">DEDQN</td> 
   <td class>3.369e+1<br>(9.025e+0)</td> 
   <td class>3.060</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.152e+4<br>(9.820e+3)</td> 
   <td class>9.498</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.097e+3<br>(4.189e+1)</td> 
   <td class>1.730</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>4.220e+3<br>(2.509e+3)</td> 
   <td class>22.580</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.763e+3<br>(2.220e+3)</td> 
   <td class>135.673</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>4.101e+0<br>(1.790e+0)</td> 
   <td class>8.153</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>5.242e+0<br>(4.506e+0)</td> 
   <td class>-47.437</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.587e-1<br>(1.185e-1)</td> 
   <td class>6.178</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">RL_HPSDE</td> 
   <td class>1.118e-1<br>(6.738e-2)</td> 
   <td class>0.010</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>4.248e+1<br>(2.126e+1)</td> 
   <td class>0.018</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>6.269e+2<br>(3.087e+2)</td> 
   <td class>0.985</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>2.820e+2<br>(1.388e+2)</td> 
   <td class>-1.257</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>6.920e+1<br>(7.215e+1)</td> 
   <td class>3.315</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>5.030e-1<br>(2.177e-1)</td> 
   <td class>0.487</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>1.203e-1<br>(1.447e-1)</td> 
   <td class>0.321</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
   <td class>6.111e-2<br>(2.326e-2)</td> 
   <td class>1.146</td> 
   <td class>2.025e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">LDE</td> 
   <td class>8.127e-9<br>(1.628e-9)</td> 
   <td class>0.000</td> 
   <td class>9.685e+3<br>(3.920e+2)</td> 
   <td class>2.681e+0<br>(2.639e+0)</td> 
   <td class>0.001</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.292e+2<br>(2.007e+2)</td> 
   <td class>0.196</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.063e+1<br>(3.898e+1)</td> 
   <td class>-2.597</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>8.336e+0<br>(6.508e+0)</td> 
   <td class>0.324</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.135e-1<br>(5.105e-2)</td> 
   <td class>-0.343</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.510e-2<br>(1.196e-2)</td> 
   <td class>1.302</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.416e-2<br>(6.351e-3)</td> 
   <td class>0.205</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">QLPSO</td> 
   <td class>2.568e+0<br>(2.915e+0)</td> 
   <td class>0.233</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.467e+2<br>(2.788e+2)</td> 
   <td class>0.064</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.411e+2<br>(2.387e+2)</td> 
   <td class>0.215</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.660e+2<br>(1.765e+2)</td> 
   <td class>-1.959</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>8.261e+0<br>(9.389e+0)</td> 
   <td class>0.320</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>5.357e-1<br>(2.517e-1)</td> 
   <td class>0.557</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.544e-2<br>(3.054e-2)</td> 
   <td class>1.205</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>5.653e-2<br>(1.812e-2)</td> 
   <td class>1.030</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">RLEPSO</td> 
   <td class>7.521e-6<br>(1.163e-5)</td> 
   <td class>0.000</td> 
   <td class>2.000e+4<br>(1.707e+0)</td> 
   <td class>7.244e+0<br>(2.131e+0)</td> 
   <td class>0.003</td> 
   <td class>2.000e+4<br>(1.473e+0)</td> 
   <td class>1.580e+2<br>(1.824e+2)</td> 
   <td class>0.241</td> 
   <td class>2.000e+4<br>(1.695e+0)</td> 
   <td class>1.139e+2<br>(1.220e+2)</td> 
   <td class>-2.274</td> 
   <td class>2.000e+4<br>(7.660e-1)</td> 
   <td class>8.816e+0<br>(1.329e+1)</td> 
   <td class>0.348</td> 
   <td class>2.000e+4<br>(7.838e-1)</td> 
   <td class>1.669e-1<br>(1.849e-1)</td> 
   <td class>-0.229</td> 
   <td class>2.000e+4<br>(8.393e-1)</td> 
   <td class>1.557e-2<br>(1.336e-2)</td> 
   <td class>1.297</td> 
   <td class>2.000e+4<br>(7.430e-1)</td> 
   <td class>1.934e-2<br>(6.975e-3)</td> 
   <td class>0.083</td> 
   <td class>2.000e+4<br>(6.044e-1)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">RL_PSO</td> 
   <td class>2.419e+0<br>(1.277e+0)</td> 
   <td class>0.220</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.607e+2<br>(1.164e+2)</td> 
   <td class>0.070</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.440e+2<br>(1.843e+2)</td> 
   <td class>0.219</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.486e+2<br>(2.018e+2)</td> 
   <td class>-1.459</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.111e+1<br>(1.222e+1)</td> 
   <td class>0.461</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>3.608e-1<br>(2.427e-1)</td> 
   <td class>0.184</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.711e-2<br>(1.978e-2)</td> 
   <td class>1.283</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.034e-2<br>(6.145e-3)</td> 
   <td class>0.108</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">DEAP_DE</td> 
   <td class>8.010e-9<br>(1.636e-9)</td> 
   <td class>0.000</td> 
   <td class>4.601e+3<br>(1.794e+2)</td> 
   <td class>5.719e+0<br>(1.373e+0)</td> 
   <td class>0.002</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.373e+2<br>(2.041e+2)</td> 
   <td class>0.209</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>3.353e+1<br>(2.353e+1)</td> 
   <td class>-2.761</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>7.477e+0<br>(8.620e+0)</td> 
   <td class>0.282</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>4.065e-2<br>(4.704e-2)</td> 
   <td class>-0.498</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.645e-2<br>(1.836e-2)</td> 
   <td class>1.289</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.581e-2<br>(7.485e-3)</td> 
   <td class>0.247</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">JDE21</td> 
   <td class>5.302e-9<br>(2.477e-9)</td> 
   <td class>-0.000</td> 
   <td class>6.283e+3<br>(1.562e+3)</td> 
   <td class>6.212e+0<br>(5.923e+0)</td> 
   <td class>0.002</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
   <td class>2.227e+2<br>(2.440e+2)</td> 
   <td class>0.344</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
   <td class>5.185e+1<br>(5.255e+1)</td> 
   <td class>-2.650</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
   <td class>1.392e+1<br>(1.440e+1)</td> 
   <td class>0.599</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
   <td class>1.122e-1<br>(9.660e-2)</td> 
   <td class>-0.345</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
   <td class>2.341e-2<br>(2.306e-2)</td> 
   <td class>1.224</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
   <td class>3.056e-2<br>(9.061e-3)</td> 
   <td class>0.368</td> 
   <td class>2.001e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">MadDE</td> 
   <td class>7.988e-9<br>(1.602e-9)</td> 
   <td class>0.000</td> 
   <td class>1.945e+4<br>(1.438e+2)</td> 
   <td class>5.361e+0<br>(1.083e+0)</td> 
   <td class>0.002</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.613e+2<br>(2.382e+2)</td> 
   <td class>0.247</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.955e+1<br>(4.236e+1)</td> 
   <td class>-2.543</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>8.824e+0<br>(9.656e+0)</td> 
   <td class>0.348</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.738e-1<br>(5.704e-2)</td> 
   <td class>-0.214</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.818e-2<br>(1.865e-2)</td> 
   <td class>1.273</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.265e-2<br>(6.212e-3)</td> 
   <td class>0.167</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">NL_SHADE_LBC</td> 
   <td class>7.826e-9<br>(1.494e-9)</td> 
   <td class>0.000</td> 
   <td class>1.444e+4<br>(1.161e+2)</td> 
   <td class>4.541e+0<br>(8.124e-1)</td> 
   <td class>0.001</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.849e+2<br>(2.153e+2)</td> 
   <td class>0.284</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.477e+1<br>(1.638e+1)</td> 
   <td class>-2.874</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>9.374e+0<br>(8.707e+0)</td> 
   <td class>0.375</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>7.150e-2<br>(4.108e-2)</td> 
   <td class>-0.432</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.734e-2<br>(1.887e-2)</td> 
   <td class>1.281</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.379e-2<br>(7.267e-3)</td> 
   <td class>0.196</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">DEAP_PSO</td> 
   <td class>1.906e+0<br>(8.624e-1)</td> 
   <td class>0.173</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.679e+2<br>(5.578e+2)</td> 
   <td class>0.118</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.950e+2<br>(2.453e+2)</td> 
   <td class>0.459</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.381e+2<br>(1.093e+2)</td> 
   <td class>-2.128</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.455e+1<br>(1.551e+1)</td> 
   <td class>0.630</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.121e-1<br>(1.315e-1)</td> 
   <td class>-0.132</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.024e-2<br>(3.588e-2)</td> 
   <td class>1.254</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.935e-2<br>(1.211e-2)</td> 
   <td class>0.337</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">GL_PSO</td> 
   <td class>1.292e-6<br>(7.497e-7)</td> 
   <td class>0.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.867e+0<br>(8.194e-1)</td> 
   <td class>0.003</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>3.322e+2<br>(3.195e+2)</td> 
   <td class>0.518</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.970e+1<br>(2.357e+1)</td> 
   <td class>-2.784</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>7.892e+0<br>(9.594e+0)</td> 
   <td class>0.302</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>4.855e-2<br>(4.491e-2)</td> 
   <td class>-0.481</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.574e-2<br>(1.518e-2)</td> 
   <td class>1.296</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.393e-2<br>(7.780e-3)</td> 
   <td class>0.199</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">sDMS_PSO</td> 
   <td class>1.661e+0<br>(7.380e-1)</td> 
   <td class>0.151</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>1.080e+2<br>(4.728e+1)</td> 
   <td class>0.047</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>1.395e+2<br>(1.922e+2)</td> 
   <td class>0.212</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>2.577e+2<br>(1.268e+2)</td> 
   <td class>-1.404</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>1.046e+1<br>(1.154e+1)</td> 
   <td class>0.429</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>3.144e-1<br>(1.308e-1)</td> 
   <td class>0.085</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>1.553e-2<br>(1.463e-2)</td> 
   <td class>1.298</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
   <td class>2.114e-2<br>(6.981e-3)</td> 
   <td class>0.128</td> 
   <td class>2.010e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">SAHLPSO</td> 
   <td class>4.477e+0<br>(2.875e+0)</td> 
   <td class>0.407</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>9.651e+2<br>(9.798e+2)</td> 
   <td class>0.426</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>4.189e+2<br>(2.584e+2)</td> 
   <td class>0.655</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>5.212e+2<br>(4.000e+2)</td> 
   <td class>0.191</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.622e+1<br>(2.821e+1)</td> 
   <td class>0.711</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.077e-1<br>(3.880e-1)</td> 
   <td class>0.710</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>3.234e-2<br>(3.775e-2)</td> 
   <td class>1.141</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>3.536e-2<br>(1.426e-2)</td> 
   <td class>0.490</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">DEAP_CMAES</td> 
   <td class>7.689e-9<br>(1.844e-9)</td> 
   <td class>0.000</td> 
   <td class>4.958e+3<br>(1.931e+2)</td> 
   <td class>1.147e+0<br>(2.696e+0)</td> 
   <td class>0.000</td> 
   <td class>1.624e+4<br>(2.174e+3)</td> 
   <td class>5.743e+0<br>(4.048e+1)</td> 
   <td class>0.000</td> 
   <td class>6.119e+3<br>(3.629e+3)</td> 
   <td class>4.896e+2<br>(7.183e+2)</td> 
   <td class>0.000</td> 
   <td class>1.996e+4<br>(3.050e+2)</td> 
   <td class>1.740e+0<br>(2.708e+0)</td> 
   <td class>0.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.743e-1<br>(9.447e-1)</td> 
   <td class>0.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.547e-1<br>(2.138e-1)</td> 
   <td class>-0.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>1.610e-2<br>(5.489e-3)</td> 
   <td class>0.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">Random_search</td> 
   <td class>1.101e+1<br>(2.739e+0)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.267e+3<br>(1.023e+3)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.363e+2<br>(2.709e+2)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>6.549e+2<br>(2.107e+2)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>2.209e+1<br>(2.151e+1)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>7.437e-1<br>(3.032e-1)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>4.746e-2<br>(5.542e-2)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
   <td class>5.537e-2<br>(1.818e-2)</td> 
   <td class>1.000</td> 
   <td class>2.000e+4<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">BayesianOptimizer</td> 
   <td class>9.144e-2<br>(1.197e-1)</td> 
   <td class>0.008</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>4.506e+3<br>(1.303e+4)</td> 
   <td class>1.988</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>1.024e+3<br>(2.678e+1)</td> 
   <td class>1.615</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>6.411e+3<br>(4.505e+3)</td> 
   <td class>35.841</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>4.903e+3<br>(4.360e+3)</td> 
   <td class>240.826</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>5.577e+0<br>(2.785e+0)</td> 
   <td class>11.297</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>1.351e+1<br>(1.428e+1)</td> 
   <td class>-124.559</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>4.987e-1<br>(2.792e-1)</td> 
   <td class>12.288</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
  </tr> 
  <tr height="19"> 
   <td class="xl63">RNN-OI</td> 
   <td class>6.940e+1<br>(6.824e-1)</td> 
   <td class>6.304</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>3.177e+4<br>(6.347e+2)</td> 
   <td class>14.021</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>1.099e+3<br>(1.198e+1)</td> 
   <td class>1.734</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>1.696e+4<br>(7.810e+3)</td> 
   <td class>99.712</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>3.876e+3<br>(3.552e+3)</td> 
   <td class>190.378</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>6.529e+0<br>(3.053e+0)</td> 
   <td class>13.326</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>7.524e+0<br>(5.715e+0)</td> 
   <td class>-68.720</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
   <td class>7.031e-2<br>(2.748e-2)</td> 
   <td class>1.380</td> 
   <td class>1.000e+2<br>(0.000e+0)</td> 
  </tr> <!--EndFragment--> 
 </tbody>
</table>
</body>
