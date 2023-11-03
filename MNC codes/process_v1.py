import json
from pymatgen.core.structure import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
fi=open('PBE-EigsFilsStructures-reordered.json','r')#open('PBE-EigsFils-MN1C.json','r')
fi=json.load(fi)
f=open('HSE06-EigsFils-MN1C-Co.json','r')
f=json.load(f)
g=open('MN1C_CO.json','r')
g=json.load(g)
hsemu=open('HSE06-mu.json','r')
hsemu=json.load(hsemu)
pbemu=open('fermivalues.json','r')
pbemu=json.load(pbemu)
t=fi['MN1C'].keys()#['Co']['adsorbed']['CO']['0.00V']
metals=['Co']
adsorbate=['CO', 'CO2', 'COOH', 'H', 'N2', 'N2H', 'NH3']
bias=['0.00V', '-0.50V', '-1.00V']
kpts=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
fi['MN1C']['Co']['adsorbed']['CO']['0.00V'].keys()
ct=0
#dic={'jjid':[]}
dataset=[]
for m in metals:
    for ad in adsorbate:
        for b in bias:
            #......poscar....#
            poscar=fi['MN1C'][m]['adsorbed'][ad][b]['POSCAR']

            atoms=JarvisAtomsAdaptor.get_atoms(Structure.from_dict(poscar))
            atoms=atoms.to_dict()


            nums=1
            elems=[]
            for ij1,ij2 in enumerate(atoms['elements']):
                if ij1!=0 and ij2!=atoms['elements'][ij1-1]:
                    nums=1
                elems.append(ij2+str(nums))
                nums+=1

            for kp in kpts:
                pbes=fi['MN1C'][m]['adsorbed'][ad][b][kp]['eigenvalues']
                hses=f['MN1C'][m]['adsorbed'][ad][b][kp]['eigenvalues']
#                 print(pbes,hses)
                ct=0
               # dic={}
                counter=0
                for i1,i2 in zip(pbes,hses):

                    if ((i1*27.211399)-(pbemu['MN1C'][m]['adsorbed'][ad][b]['mu']*27.21139)>-10) & ((i1*27.211399<10)-(pbemu['MN1C'][m]['adsorbed'][ad][b]['mu']*27.21139)<10):
                        dic={}
                        dic['jjid']=ad+'_'+m+'_MN1C_ads_'+b+'_bias_'+kp+'_kpt_'+str(ct)
                        hs=(i2-hsemu['MN1C'][m]['adsorbed'][ad][b]['mu'])*27.21139
                        pbs=(i1-pbemu['MN1C'][m]['adsorbed'][ad][b]['mu'])*27.21139
                        dic['shifted']=hs
                        dic['shifted_pbe']=pbs
                        dic['atoms']=poscar
                        dic['nf']=[]
                        for k4 in elems:
                            
                           nf=[]
                           for k5 in g['adsorbed'][ad][b][kp]['BandProjections'][k4]:
                               if k5=='element':
                                   continue
                               else:
                                   nf.append(g['adsorbed'][ad][b][kp]['BandProjections'][k4][k5][counter])
                           dic['nf'].append(nf)
                        dataset.append(dic)
                        ct+=1
                    counter+=1

with open('MN1C_left.json','w') as f:
    json.dump(dataset,f)
