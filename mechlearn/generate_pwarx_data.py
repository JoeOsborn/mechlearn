import testing
import mm_systems as mm
def generate_data():
    Nsamples = 700
    Ts = 1.0
    Rl=2000.0; Rc=400; C=0.3
    X0=3.5; Vin=10.0; Vmin=3.0; Vmax=4.0

    for i in range(1, 31):
        data = mm.rc_circuit(Nsamples,Ts,Rl,Rc,C,X0,Vin,Vmin,Vmax,prob=-1.0)
        yfilename = 'pwa_y_data_'+str(i)+'.txt'
        ufilename = 'pwa_u_data_'+str(i)+'.txt'
        mfilename = 'pwa_m_data_'+str(i)+'.txt'
        
        f = open(yfilename, 'w')
        for y in data.x:
            f.write(str(y[0])+'\n')
        f.close()

        f = open(ufilename, 'w')
        for u in data.u:
            f.write(str(u[0])+'\n')
        f.close()

        f = open(mfilename, 'w')
        for m in data.m:
            f.write(str(m)+'\n') 
        f.close()
