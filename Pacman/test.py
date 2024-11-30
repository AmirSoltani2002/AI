import pygame
from pygame.locals import *
import numpy as np
import time
class Env:
  def __init__(self, row, col, lamda, alpha, epsilon = 0.9, reward = 2):
    self.col = col
    self.row = row
    self.reward = np.ones((row, col))
    self.reward[0, 0] = 0
    self.Q = np.zeros((4, 8**8))
    self.lamda = lamda
    self.alpha = alpha
    self.epsilon = epsilon
    self.current_row = 0
    self.current_col = 0
    self.food = row * col
    self.curr_food = 0
    self.current_reward = 0

  def ghost_move(self):
    self.ghost_map[self.ghost_row, self.ghost_col] = 0
    row = np.random.randint(-1, 2)
    col = np.random.randint(-1, 2)
    self.ghost_row = self.ghost_row + row if 0<=(self.ghost_row + row)<self.row else self.ghost_row
    self.ghost_col = self.ghost_col + col if 0<=(self.ghost_col + col)<self.col else self.ghost_col
    self.ghost_map[self.ghost_row, self.ghost_col] = 1

  def reset_map(self):
    self.add_ghost(self.init_g_row, self.init_g_col)
    self.reward = np.ones((self.row, self.col))
    self.reward[0, 0] = 0
    self.add_wall(self.wall_pos, True)

  def add_wall(self, positions, update = False, reward = -2):
    self.wall_pos = positions
    self.reward[positions[:, 0], positions[:, 1]] = reward
    if not update:
      self.food -= (positions.shape[0] + 1)

  def add_ghost(self, row, col):
    self.ghost_map = np.zeros((self.row, self.col))
    self.init_g_row = row
    self.init_g_col = col
    self.ghost_row = row
    self.ghost_col = col
    self.ghost_map[self.ghost_row, self.ghost_col] = 1

  def state(self, row, col):#0->boundary, 1->food, 2->empty, 3->wall
    d_r = u_r = u_l = d_l = down = right = up = left = 0
    if row != self.row-1:
      if self.reward[row+1, col] == 1: down = 1
      elif self.reward[row+1, col] == 0: down = 2
      else: down = 3
      if self.ghost_map[row+1, col] == 1: down += 4
    if col != self.col-1:
      if self.reward[row, col+1] == 1: right = 1
      elif self.reward[row, col+1] == 0: right = 2
      else: right = 3
      if self.ghost_map[row, col+1] == 1: right += 4
    if row != 0:
      if self.reward[row-1, col] == 1: up = 1
      elif self.reward[row-1, col] == 0: up = 2
      else: up = 3
      if self.ghost_map[row-1, col] == 1: up += 4
    if col != 0:
      if self.reward[row, col-1] == 1: left = 1
      elif self.reward[row, col-1] == 0: left = 2
      else: left = 3
      if self.ghost_map[row, col-1] == 1: left += 4
    if down != 0 and right != 0:
      if self.reward[row+1, col+1] == 1: d_r = 1
      elif self.reward[row+1, col+1] == 0: d_r = 2
      else: d_r = 3
      if self.ghost_map[row+1, col+1] == 1: d_r += 4
    if up != 0 and right != 0:
      if self.reward[row-1, col+1] == 1: u_r = 1
      elif self.reward[row-1, col+1] == 0: u_r = 2
      else: u_r = 3
      if self.ghost_map[row-1, col+1] == 1: u_r += 4
    if up != 0 and left != 0:
      if self.reward[row-1, col-1] == 1: u_l = 1
      elif self.reward[row-1, col-1] == 0: u_l = 2
      else: u_l = 3
      if self.ghost_map[row-1, col-1] == 1: u_l += 4
    if down != 0 and left != 0:
      if self.reward[row+1, col-1] == 1: d_l = 1
      elif self.reward[row+1, col-1] == 0: d_l = 2
      else: d_l = 3
      if self.ghost_map[row+1, col-1] == 1: d_l += 4
    return (8**7*down + 8**6*right + up * 8**5 + left * 8**4 + d_r * 8**3 + u_r * 8 ** 2 + u_l * 8 + d_l)

  def check_move(self):
    for j in range(self.Q.shape[1]):
      res = self.check_state(j)
      for i in range(4):
        if res[i] == 0:
          self.Q[i, j] = -np.inf

  def check_state(self, state):
    num = state
    res = [0, 0, 0, 0, 0, 0, 0, 0]
    i = 0
    while num > 0:
      res[7 - i] = num % 8
      i += 1
      num //= 8
    return res

  def action_reward(self, state, action):
    res = self.check_state(state)
    if action == 0:
      if res[action] == 1:
        self.current_row += 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return 2
      elif res[action] == 2:
        self.current_row += 1
        return -1
      elif res[action] == 3:
        return -2
      elif res[action] == 5:
        self.current_row += 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return -40
      elif res[action] == 6:
        self.current_row += 1
        return -50
      elif res[action] == 7:
        return -5

    elif action == 1:
      if res[action] == 1:
        self.current_col += 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return 2
      elif res[action] == 2:
        self.current_col += 1
        return -1
      elif res[action] == 3:
        return -2
      elif res[action] == 5:
        self.current_col += 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return -40
      elif res[action] == 6:
        self.current_col += 1
        return -50
      elif res[action] == 7:
        return -5

    elif action == 2:
      if res[action] == 1:
        self.current_row -= 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return 2
      elif res[action] == 2:
        self.current_row -= 1
        return -1
      elif res[action]== 3:
        return -2
      elif res[action] == 5:
        self.current_row -= 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return -40
      elif res[action] == 6:
        self.current_row -= 1
        return -50
      elif res[action] == 7:
        return -5

    elif action == 3:
      if res[action] == 1:
        self.current_col -= 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return 2
      elif res[action] == 2:
        self.current_col -= 1
        return -1
      elif res[action] == 3:
        return -2
      elif res[action] == 5:
        self.current_col -= 1
        self.curr_food += 1
        self.reward[self.current_row, self.current_col] = 0
        return -40
      elif res[action] == 6:
        self.current_col -= 1
        return -50
      elif res[action] == 7:
        return -5


  def move(self, state):
    actions = self.Q[:, state]
    action = 0
    if np.random.random() < self.epsilon:
      while True:
        action = np.random.randint(0, 4)
        if actions[action] != -np.inf:
          break
    else:
      action = np.where(actions == max(actions))[0][0]
    reward = self.action_reward(state, action)
    #print(reward, self.current_reward)
    self.current_reward += reward
    next_state = self.state(self.current_row, self.current_col)
    self.Q[action, state] += self.alpha * (reward + self.lamda * max(self.Q[:, next_state]) - self.Q[action, state])

  def episod(self):
    self.current_row = 0
    self.current_col = 0
    iteration = 0
    while self.curr_food < self.food:
      self.ghost_move()
      state = self.state(self.current_row, self.current_col)
      self.move(state)
      iteration += 1
      if self.current_col == self.ghost_col and self.current_row == self.ghost_row:
        print('Loose Game!', end=', ')
        break
    return iteration

  def game(self):
    i = 0
    c = 1.01
    self.check_move()
    while i < 300:
      print('ep', i, ':', self.epsilon)
      self.reset_map()
      self.epsilon =  self.epsilon / c**i
      print('episods/reward:  ', self.episod(), '/', self.current_reward)
      self.current_reward = 0
      self.curr_food = 0
      i += 1

  def test_game(self):
    pygame.init()
    # Set up the display
    cell_size = 100  # Size of each cell in pixels
    map_width = self.col  # Number of cells in the horizontal direction
    map_height = self.row # Number of cells in the vertical direction
    screen_width = cell_size * map_width
    screen_height = cell_size * map_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pacman")
    # Define colors
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    self.reset_map()
    self.current_reward = 0
    self.curr_food = 0
    self.current_row = 0
    self.current_col = 0
    last_x = last_y = 0
    self.epsilon = 0.05
    last_food = 0
    iterate_last_food = 0
    while self.curr_food < self.food:
        self.ghost_move()
        state = self.state(self.current_row, self.current_col)
        self.move(state)
        pacman_map = self.reward
        screen.fill(BLACK)
        # Render the map
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break
        for x in range(map_height):
            for y in range(map_width):
                cell_color = WHITE
                if pacman_map[x][y] == -2:
                    cell_color = BLACK
                if pacman_map[x][y] == 1:
                    cell_color = BLUE
                rectangle = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                    #cell_color = YELLOW
                pygame.draw.rect(screen, cell_color, rectangle)
                if x == self.current_row and y == self.current_col:
                    if last_x > self.current_row:
                        image = pygame.image.load("up_pac.png")  # Replace "image.jpg" with the path to your image file
                    elif last_x < self.current_row:
                        image = pygame.image.load("bottom_pac.png") 
                    elif last_y < self.current_col:
                        image = pygame.image.load("right_pac.png") 
                    elif last_y > self.current_col:
                        image = pygame.image.load("left_pac.png") 
                    else:
                        image = pygame.image.load("right_pac.png")
                    image = pygame.transform.scale(image, (cell_size, cell_size))
                    screen.blit(image, (y * cell_size, x * cell_size))
                    pygame.display.update()
                if x == self.ghost_row and y == self.ghost_col:
                    if cell_color == WHITE:
                        image = pygame.image.load("white.png")
                    if cell_color == BLACK:
                        image = pygame.image.load("black.png")
                    if cell_color == BLUE:
                        image = pygame.image.load("blue.png")
                    image = pygame.transform.scale(image, (cell_size, cell_size))
                    # Blit the original image onto the new surface with the desired background
                    # Blit the new surface onto the screen
                    #image.fill(cell_color)
                    screen.blit(image, (y * cell_size, x * cell_size))
                    pygame.display.update()
        last_x = self.current_row
        last_y = self.current_col
        # Update the display
        pygame.display.flip()
        if self.current_col == last_food:
            iterate_last_food += 1
        else:
          iterate_last_food = 0
          last_food = self.current_col
        if iterate_last_food >= 30:
          self.epsilon = 0.5
        else:
          self.epsilon = 0
        time.sleep(0.5)
        if self.current_col == self.ghost_col and self.current_row == self.ghost_row:
            print('Loose Game!', end=', ')
            break

if __name__ == "__main__":
  game = Env(7, 9, 0.25, 0.2)
  game.add_wall(np.array([[1, 1], [1, 2], [1, 3], [2, 1], [4, 1], [5, 1], [3, 3], [4, 3], [5, 4], [4, 5], [3, 5], [1, 5], [1, 6], [1, 7], [2, 7], [4, 7], [5, 7]]))
  game.add_ghost(4, 4)
  print(game.reward)
  game.game()
  game.test_game()
  # Initialize Pygame
