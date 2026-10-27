"""Meaningful pixel-center and label-placement controls for 192-mm exports."""
import numpy as np
import nibabel as nib

from preprocessing.myopspp_unet2d_roi192 import resize_to, numpy_resize_image


def test_192_to_128_maps_pixel_centers_without_shifting_physical_fov():
    x,y=np.indices((192,192),dtype=np.float32)
    image=(2*x+3*y)[:,:,None]
    affine=np.array([[-1.,0,0,100],[0,1.,0,-20],[0,0,12.,5],[0,0,0,1.]])
    result,out_affine=resize_to(image,affine,128)
    assert result.shape==(128,128,1)
    assert np.allclose(nib.affines.voxel_sizes(out_affine),[1.5,1.5,12.])
    assert np.allclose(out_affine@[0,0,0,1],affine@[.25,.25,0,1])
    assert np.allclose(out_affine@[-.5,-.5,0,1],affine@[-.5,-.5,0,1])
    assert np.allclose(out_affine@[127.5,127.5,0,1],affine@[191.5,191.5,0,1])
    assert np.isclose(result[0,0,0],1.25)
    assert np.allclose(result,numpy_resize_image(image,128),atol=1e-5)


def test_nearest_exact_192_to_128_matches_independent_indices_and_preserves_z():
    codes=np.array([0,200,500,600,1220,2221],dtype=np.uint16)
    x,y=np.indices((192,192))
    label=np.stack([codes[(x+2*y+z)%len(codes)] for z in range(3)],axis=2)
    result,_=resize_to(label,np.diag([1.,1.,10.,1.]),128,True)
    indices=np.floor((np.arange(128)+.5)*1.5).astype(int)
    expected=label[indices[:,None],indices[None,:],:]
    assert result.dtype==np.uint16
    assert np.array_equal(result,expected)
    assert set(np.unique(result)).issubset(set(codes))


def test_192_identity_keeps_array_values_and_affine_without_resampling():
    a=np.zeros((192,192,2),dtype=np.uint16);a[5,7,1]=2221
    affine=np.diag([-1.,1.,8.,1.]);affine[:3,3]=[12.,-15.,7.]
    result,result_affine=resize_to(a,affine,192,True)
    assert np.array_equal(a,result)
    assert np.array_equal(affine,result_affine)
    assert result_affine is not affine
