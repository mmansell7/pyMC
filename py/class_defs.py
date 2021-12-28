# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5


import numpy as np
from scipy.spatial import distance_matrix

mcdtypes = {'atomtypename':'U20',
            'atomtypemass':'f8',
            'atomtag':'i8',
            'atomtypeindex':'i4',
            'atomx':'f8',
            'atome':'f8'}

#%%
class MC:
    
    def __init__(self,seed):
        
        self.rng = np.random.default_rng(seed)
        self.geom = [1.0,2.0,3.0]
        self.atomtypename  = np.array([],dtype=mcdtypes['atomtypename'])
        self.atomtypemass  = np.array([],dtype=mcdtypes['atomtypemass'])
        self.atomtag       = np.array([],dtype=mcdtypes['atomtag'])
        self.atomtypeindex = np.array([],dtype=mcdtypes['atomtypeindex'])
        self.atomx         = np.empty((0,3),dtype=mcdtypes['atomtypeindex'])
        self.atome         = np.array([],dtype=mcdtypes['atome'])

    def distance(self,i,j):
        d = self.atomx[j] - self.atomx[i]
        d = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
        return d
    
    def pair_potential(self,i,j):
        d = self.distance(i,j)
        v = abs(d-1.0)
        return v
    
    def calc_V(self):
        for i in range(0,num_atoms):
            for j in range(i+1,num_atoms):
                v = mc.pair_potential(i,j)
                mc.atome[i] += v
                mc.atome[j] += v
#%%
class Pointers:
    
    def __init__(self,mc):
        self.mc = mc
        return
    
    @property
    def geom(self):
        return self.mc.geom
    
    @property
    def rng(self):
        return self.mc.rng
    
    @property
    def atomtype(self):
        return self.mc.atomtype
    
    @property
    def atom(self):
        return self.mc.atom
    
    
#%%
class PairList(Pointers):
    
    def __init__(self,mc):
        Pointers.__init__(self,mc)
        
#%%
if __name__ == '__main__':
    num_atoms = 10
    print('Hello, world!')
    mc = MC(12345)
    print(mc)
    for atyp in [('Cfixed',12),('Cmobile',12),('Ar',40)]:
        mc.atomtypename = np.append(mc.atomtypename,atyp[0])
        mc.atomtypemass = np.append(mc.atomtypemass,atyp[1])
    print('mc.atomtypename: {}'.format(mc.atomtypename))
    print('mc.atomtypemass: {}'.format(mc.atomtypemass))
    mc.atomtag = np.append(mc.atomtag,np.arange(num_atoms,dtype=mcdtypes['atomtag']))
    print('mc.atomtag: {}'.format(mc.atomtag))
    mc.atomtypeindex = np.append(mc.atomtypeindex,np.zeros(num_atoms,dtype=mcdtypes['atomtypeindex']))
    print('mc.atomtypeindex: {}'.format(mc.atomtypeindex))
    mc.atomx = np.append(mc.atomx,mc.rng.random((num_atoms,3)),axis=0)
    print('mc.atomx: {}'.format(mc.atomx))
    print('mc.atome: {}'.format(mc.atome))
    
    print('Done. Exiting.')
    
