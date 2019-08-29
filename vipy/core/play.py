import pygame
from PIL import Image
import multiprocessing
import sys

# Window state
IMG = None
SCREEN = None
DRAWQUEUE = multiprocessing.Queue()


def imshow(imgfile, title):
    DRAWQUEUE.put(('_imshow', (imgfile, title)))

def rectangle(bbox, caption):
    DRAWQUEUE.put(('_rectangle', (bbox, caption)))
    

def tracking(instream, framerate=None):
    # Initialize window
    pygame.init()
    pygame.font.init()
    (im, anno) = instream(async=True)[0]
    imsize = (im.get().shape[0], im.get().shape[1])
    screen = pygame.display.set_mode(imsize) 
    pygame.display.set_caption('tracking')
    imshown = im.url()
    img = pygame.image.load(im.url()) 

    # Initialize text
    #font = pygame.font.SysFont(None, 12)
    font = pygame.font.Font(None, 12)

    # Update display
    for (im, anno) in instream(async=True):
        #print anno        
        if anno['imurl'] != imshown:
            screen.blit(img, (0,0))
            if framerate is not None:
                pygame.time.wait(int(1000*(1.0/framerate)))
            pygame.display.flip() # update the display            
            img = pygame.image.load(im.url())             
            imshown = im.url()
        bbox = (anno['bbox_xmin'], anno['bbox_ymin'], anno['bbox_xmax']-anno['bbox_xmin'], anno['bbox_ymax']-anno['bbox_ymin'])
        pygame.draw.rect(img, (255,0,0), bbox, 1)

        text = font.render('%d' % anno['trackid'], 1, (0, 255, 0))
        textrect = text.get_rect()
        textrect.centerx = bbox[0] 
        textrect.centery = bbox[1]
        img.blit(text, textrect)
        
        #bbox(, imshape=(im.shape[0], im.shape[1]), bboxcaption=anno['trackid'])




def _imshow(imgfile, title=None):
    global IMG
    global SCREEN

    im = Image.open(imgfile)  # do not load pixel buffer, just get size
    SCREEN = pygame.display.set_mode(im.size) 
    if title is not None:
        pygame.display.set_caption(title)
    IMG = pygame.image.load(imgfile) 
    SCREEN.blit(IMG, (0,0))
    pygame.display.flip() # update the display                

def _rectangle(bbox, caption=None):
    global IMG
    global SCREEN
    
    font = pygame.font.SysFont(None, 12)
    pygame.draw.rect(IMG, (255,0,0), bbox, 1)
    SCREEN.blit(IMG, (0,0))
    text = font.render('%s' % caption, 1, (0, 255, 0))
    textrect = text.get_rect()
    textrect.centerx = bbox[0] 
    textrect.centery = bbox[1]
    IMG.blit(text, textrect)
    pygame.display.flip() # update the display                
    

def _fullscreen():
    pygame.display.toggle_fullscreen()

def _close():
    pygame.display.quit()

    

def _mainloop(drawqueue):
    def _mainloop_(drawqueue):
        import pygame
        pygame.init()

        Clock = pygame.time.Clock()
        done = False
        while not done:
            # GUI events
            Clock.tick(30)
            #pygame.display.set_caption("Press Esc to quit. FPS: %.2f" % (Clock.get_fps()))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True # Be IDLE friendly!
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True # Be IDLE friendly!

            # Drawing events
            if not drawqueue.empty():
                (funcname, args) = drawqueue.get()
                func = globals()[funcname]
                func(*args)
                        
        print 'quitting'
        sys.stdout.flush()
        pygame.display.quit()

    p = multiprocessing.Process(target=_mainloop_, args=(drawqueue,))
    p.daemon = True
    p.start()        

    
# Execute mainloop!
_mainloop(DRAWQUEUE)
