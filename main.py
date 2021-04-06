import pygame, sys
import joblib
import matplotlib.pyplot as plt
import numpy
from pygame.locals import *
import warnings
from PIL import  Image

from keras.models import load_model

warnings.filterwarnings('ignore')


def main():
    pygame.init()
    num = ''
# Initialize all required gui

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0,255,0)
    BLUE  = (0,0,255)
    RED   = (255,0,0)

    mouse_position = (0, 0)
    drawing = False
    screen = pygame.display.set_mode((960,560), 0, 32)
    screen.fill(WHITE)
    pygame.display.set_caption("Draw Digits")
    PIXEL = numpy.empty((224,224,3))

    #Title
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render('MNIST Digit Recognizer', True,BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (480, 50)

    #Input Text
    fontSm = pygame.font.Font('freesansbold.ttf', 16)
    textIn = fontSm.render('Draw your digit below (between 0-9) and Please draw it big enough', True,BLACK, WHITE)
    textInRect = textIn.get_rect()
    textInRect.center = (480, 100)



#Functions

    #Clear the input after draw

    def clear():
        screen.fill(WHITE)


    #Draw Input
    def draw_rect(num):
        textOut = font.render('Predicted result: {}'.format(num), True, BLACK, WHITE)
        textOutRect = textOut.get_rect()
        textOutRect.center = (240, 240)
        pygame.draw.rect(screen,BLACK,pygame.Rect(600,240,240,240),4)
        screen.blit(text, textRect)
        screen.blit(textIn, textInRect)
        screen.blit(textOut, textOutRect)


    #Retry Button
    def retry():
        button = font.render("Retry",True,GREEN)
        button = screen.blit(button,[130,410])
        return button


    #Model Prediction

    def standardize(pixel):
        pixel = numpy.abs(pixel-255)
        # pixel =pixel/255
        return pixel

    def rgb2gray(rgb):
        rgb = numpy.rot90(rgb,k=1)
        rgb = numpy.flipud(rgb)
        return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def get_model():
        model = load_model('cnn_digits.h5')
        return model

    def get_pred(m,p):
        pred = m.predict(p)
        return pred


    def print_pred(P):
        img = standardize(P)
        img = rgb2gray(img)
        img = Image.fromarray(img)
        img = img.resize(size=(28,28))
        img = numpy.array(img)
        img = img.reshape(1, 28, 28, 1)
        pred = get_pred(model, img)
        return numpy.nanargmax(pred)



    model = get_model()




    while True:

        draw_rect(num)
        button = retry()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                if (drawing):
                    mouse_position = pygame.mouse.get_pos()
                    m_X,m_Y =mouse_position
                    if (m_X>605 and m_X<835) and (m_Y>245 and m_Y<475):
                        pygame.draw.circle(screen, BLACK, mouse_position, 10)

            elif event.type == MOUSEBUTTONUP:
                drawing = False
            elif event.type == MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                drawing = True
                if button.collidepoint(mouse_position):
                    clear()
            elif event.type == pygame.KEYDOWN:

                if event.key ==K_SPACE:
                    for i in range(224):
                        for j in range(224):
                            PIXEL[i][j]=list(screen.get_at([i+604,j+244])[:3])
                    num = str(print_pred(PIXEL))
                    clear()


        pygame.display.update()

if __name__ == "__main__":
    main()