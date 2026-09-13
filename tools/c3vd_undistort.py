"""Undistort C3VD colonoscope frames from the Scaramuzza omnidirectional model to a pinhole camera.

C3VD is captured with a cfhq190l colonoscope -- a ~190-degree fisheye. 3D Gaussian Splatting renders
a pinhole camera, so raw C3VD frames cannot be used directly: fitting a pinhole model to them fails
badly (verified: reprojection error 70+ px against a 6 px optical-flow baseline).

Calibration comes from the dataset's own cfhq190l_omnidirectional_params.zip:
    width height cx cy a0 a2 a3 a4 e f g
where the radial polynomial is  z(rho) = a0 + a2*rho^2 + a3*rho^3 + a4*rho^4  and [e f; g 1] is the
affine stretch. A pixel at radius rho from the centre corresponds to ray angle atan2(rho, z(rho));
we invert that numerically to build the pinhole -> fisheye sampling map.

The lens covers a 103.6 degree half-FOV; undistorting much beyond ~100 degrees full-FOV stretches the
corners severely, so keep fov_deg modest.

STATUS: undistortion verified visually. The dataset poses do NOT yet reproject consistently
(best fit 9.07 px vs a 6.13 px no-motion baseline), so the pose convention/scale is unresolved --
do not train on this until that check passes.
"""
import numpy as np, cv2
W,H,cx,cy = 1350,1080,679.544839263292,543.975887548343
A=[769.243600037458,0.0,-0.000812770624150226,6.25674244578925e-07,-1.19662182144280e-09]  # a0,a1,a2,a3,a4
e,f_,g = 0.999986882249990,0.00288273829525059,-0.00296316513429569

def poly(rho): return A[0]+A[2]*rho**2+A[3]*rho**3+A[4]*rho**4
# table: theta (angle from optical axis) -> rho (pixels from centre)
rho=np.linspace(0,900,4000)
z=poly(rho)
theta=np.arctan2(rho,z)              # 0 at centre, grows outward
ok=np.isfinite(theta)&(np.diff(np.r_[theta,theta[-1]])>=0)
theta_t,rho_t=theta[ok],rho[ok]
print('max theta covered: %.1f deg (half-FOV)'%np.degrees(theta_t.max()))

def undistort(img, fov_deg=120.0, out=(960,768), interp=cv2.INTER_LINEAR):
    ow,oh=out; fv=(ow/2)/np.tan(np.radians(fov_deg)/2)
    ys,xs=np.mgrid[0:oh,0:ow].astype(np.float64)
    dx,dy=xs-ow/2,ys-oh/2
    r=np.hypot(dx,dy); th=np.arctan2(r,fv)
    rho_q=np.interp(th,theta_t,rho_t,left=0,right=np.nan)
    scale=np.divide(rho_q,np.maximum(r,1e-9))
    u=dx*scale; v=dy*scale
    # inverse affine (stretch matrix [e f; g 1])
    xs_src = u*e + v*f_ + cx
    ys_src = u*g + v*1.0 + cy
    m=np.isfinite(xs_src)&np.isfinite(ys_src)
    xs_src=np.where(m,xs_src,-1).astype(np.float32); ys_src=np.where(m,ys_src,-1).astype(np.float32)
    return cv2.remap(img,xs_src,ys_src,interp,borderValue=0), fv

D='data/desc_t4_a'
c=cv2.imread(f'{D}/0000_color.png')
u120,fv=undistort(c,120.0)
print('virtual pinhole focal for 120deg on 960px: %.1f'%fv)
cv2.imwrite('/tmp/undist_cmp.png', np.hstack([cv2.resize(c,(960,768)), u120]))
