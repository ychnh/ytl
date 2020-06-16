import time
import datetime

class pgrs:
    def __init__(self, I):
        self.DISP_LEN = 15
        self.L = I
        self.I = iter(I)
        
        
    def __next__(self):
        D, L, ta, tb = self.DISP_LEN, len(self.L), self.ta, time.time()
        

        p = self.i/L
        P = int( D*p )
        per = str( int(100*p) )
        per = ( per+'%').rjust(4,' ')
        prg = (P)*'█'+ (D-P)*' '
        
        
        t_cnt = str(L)
        c_cnt = str(self.i)
        c_cnt = c_cnt.rjust( len(t_cnt) )
        
        t = tb-ta
        avg_t = (self.elapsed_t+t)/(self.i+1)
        
        sec_remain = int( avg_t*(L-self.i) + .9 )
        str_remain = str(datetime.timedelta(seconds=sec_remain))
        str_avgt = ' '+str( round(avg_t,4) )+'_itr'
        
        print(per+'|'+prg+'| '+c_cnt+'/'+t_cnt + ' [' + str_remain +' | '+str_avgt+']', end='\r')
        
        self.ta = time.time()
        self.elapsed_t += t
        self.i+=1
        return next(self.I)
    
    def __len__(self):
        return len(self.I)
    
    def __iter__(self):
        self.ta = time.time()
        self.elapsed_t = 0
        self.i = 0
        return self
