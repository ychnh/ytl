red = [255,0,0]
green = [0,255,0]
blue = [0,0,255]

def point( src, r,c, clr):
    src[r,c] = clr

# Note all the other drawing functions are abstracted based on the single function, draw point.
# If the data structure is different, simply redefine the point drawing algorithm - yh

def line(src, r,c, l, dir, clr):
    if dir=='row':
        for i in range(l):
            point(src, r+i,c, clr)
    elif dir=='col':
        for i in range(l):
            point(src, r,c+i, clr)

def bbox(src, r,c, lr, lc, clr):
    r_,c_ = r+lr, c+lc
    line(src, r,c, lr, 'row', clr)
    line(src, r,c_, lr, 'row', clr)
    line(src, r,c, lc, 'col', clr)
    line(src, r_,c, lc+1, 'col', clr)

def square( src, r,c, clr):
    point(src, r-1,c-1, clr)
    point(src, r-1,c, clr)
    point(src, r-1,c+1, clr)
    point(src, r,c-1, clr)
    point(src, r,c, clr)
    point(src, r,c+1, clr)
    point(src, r+1,c-1, clr)
    point(src, r+1,c, clr)
    point(src, r+1,c+1, clr)
