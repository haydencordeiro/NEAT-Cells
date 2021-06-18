# from main import run
from genericpath import samefile
import pygame
import os
import random
import math
import sys
import neat
import time

class StoreData(object):

    savedTime= time.time()
    valueright=None
    valuebottom=None
    def StoreValue(self,right,bottom):
        if  time.time()-self.savedTime>=5:
            if (self.valueright!=None and self.valueright==right and self.valuebottom==bottom ):
                
                return False
            else:
                print(self.valueright,self.valuebottom,right,bottom)
            
            self.savedTime=time.time()
            self.valueright=right
            self.valuebottom=bottom
            return True
        else:
            return True


pygame.init()
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
size = (700, 500)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
pygame.display.set_caption("My First Game")
FONT = pygame.font.Font('freesansbold.ttf', 20)

class Player(object):
    def __init__(self,color):
        self.rect = pygame.rect.Rect((64, 54, 8, 8))
        self.color=color

        
    def handle_keys(self,val):
        text_1 = FONT.render(f'Players Alive:  {str(len(players))}', True, (0, 0, 0))
        text_2 = FONT.render(f'Generation:  {pop.generation+1}', True, (0, 0, 0))
        screen.blit(text_1, (50, 400))
        screen.blit(text_2, (50, 450))
        if(self.rect.right>=700 and self.rect.bottom>=500):
            return "Reached Goal"
        if(self.rect.right<0 or self.rect.bottom<0):
            return False
        dist = 1
        if val[0]==1:#left
           self.rect.move_ip(-dist, 0)
        if val[1]==1:#right
           self.rect.move_ip(dist, 0)
        if val[2]==1:#up
            self.rect.move_ip(0, -dist)
        if val[3]==1:#down
           self.rect.move_ip(0, dist)

        return True

    def draw(self, surface):
        pygame.draw.rect(screen, self.color, self.rect)
    #rect.left,top,bottom,rgight
players = []
ge = []
nets = []
sd=[]
def eval_genomes(genomes, config):
    global players, ge, nets,sd
    clock = pygame.time.Clock()



    for genome_id, genome in genomes:
        players.append(Player(random_color()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        sd.append(StoreData())
        genome.fitness = 0
        
    running = True
 
    while running:
        # print(players[0].rect,sd[0].value)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        for i,player in enumerate(players):
            player.draw(screen)
            # print((player.rect.right,player.rect.bottom,700,500))
            output = nets[i].activate((player.rect.right,player.rect.bottom,700,500))
            # print(output)
            temp=player.handle_keys(output)
        
            if(temp==False):
                ge[i].fitness -= 1
                players.pop(i)
                ge.pop(i)
                nets.pop(i)
                sd.pop(i)
            elif(temp=="Reached Goal"):
                # print("Reached Goal")
                # print(len(ge),len(nets),len(players))
                pass
        
            elif(sd[i].StoreValue(player.rect.right,player.rect.bottom)==False):
                    ge[i].fitness -= 1
                    players.pop(i)
                    ge.pop(i)
                    nets.pop(i)
                    sd.pop(i)
                    # print("ASdf")
        if len(players) == 0:
            break 
        pygame.display.update()

        clock.tick(40)



# Setup the NEAT Neural Network
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
