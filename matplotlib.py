# Visual

def pyplot_show_grid(images):
    ''' Displays multiple `images` in a grid shape.
    * images: a list of numpy/PIL images
    '''
    L = len(images)
    R = math.floor( math.sqrt(L) )
    C = math.ceil(L/R)
    print('diplaying',L, 'images:',R,'x',C)

    plt.figure(figsize=(12,12))
    idx = 0
    f, axarr = plt.subplots(R,C)
    for i in range(R):
        for j in range(C):
            arrIdx = C*i+j
            if (arrIdx >= L):
                break
            axarr[i,j].imshow(images[R*i+j])
            idx+=1

    plt.show()

def pyplot_show_list(images):
    ''' Shows multiple large images in a single column
    * images: list of numpy/PIL images
    '''
    N = len(images)
    plt.figure(figsize=(10,25))
    for i,img in enumerate(images):
        axis = plt.subplot(N,1,i+1)
        plt.imshow(img)
        
def pyplot_histogram(X, bins, title):                                                                                                                                            
    plt.hist(X, normed=True, bins=bins)                                                                                                                                          
    plt.ylabel('Title')  
