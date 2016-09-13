import h5py
import numpy as np

def dump_h5(filename, prefix, data):
    ''' Dumps data to a new dataset or
    appends to the existed dataset 
    
    '''
    h5f = h5py.File(filename, 'a')

    ds = h5f.get(prefix)
    data = np.asarray(data)

    if ds:
        offset = len(ds)
        ds.resize(len(ds) + len(data), axis=0)
        for i in range(len(data)):
            ds[offset+i] = data[i]
    else:
        ds = h5f.create_dataset(prefix, maxshape=(None,)+data.shape[1:],
            data=data)

    h5f.close() 

def dump_h5_var(filename, prefix, prefix_shape, data):
    ''' Dumps variable length data to a new dataset or
    appends to the existed dataset 
    
    '''
    h5f = h5py.File(filename, 'a')

    ds = h5f.get(prefix)
    ds_shp = h5f.get(prefix_shape)

    if not ds:
        var_dt = h5py.special_dtype(vlen=np.dtype(data[0].dtype))
        ds = h5f.create_dataset(prefix, shape=(len(data),), maxshape=(None,), dtype=var_dt)
        dim = len(data[0].shape)
        ds_shp = h5f.create_dataset(prefix_shape, shape=(len(data),dim), maxshape=(None,dim), dtype=np.int64)
        offset = 0
        offset_shp = 0

    else:
        offset = len(ds)
        offset_shp = len(ds)

        ds.resize(len(ds) + len(data), axis=0)
        ds_shp.resize(len(ds_shp) + len(data), axis=0)

    for i in range(len(data)):
        ds[offset+i] = data[i].flatten()
        ds_shp[offset_shp+i] = data[i].shape

    h5f.close() 


