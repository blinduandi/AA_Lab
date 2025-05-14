import pygame
import random
import sys
from collections import deque

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 150, 0)
LIME_GREEN = (50, 205, 50)
RED = (200, 0, 0)
PINK = (255, 105, 180)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRS = (UP, DOWN, LEFT, RIGHT)

# Game settings
FPS = 10
INITIAL_SNAKE_LENGTH = 3
BONUS_FOOD_CHANCE = 0.2
BONUS_FOOD_SCORE = 50
NORMAL_FOOD_SCORE = 10

# UI
BUTTON_RECT = pygame.Rect(SCREEN_WIDTH - 140, 10, 130, 40)


# --- Helper utilities ----------------------------------------------------

def next_pos_after_move(pos, direction):
    """Return grid position after moving one step (handling teleporters)."""
    x, y = pos[0] + direction[0], pos[1] + direction[1]
    return x, y


def teleport_if_needed(pos, board):
    return board.teleporters.get(pos, pos)


# --- Game Classes --------------------------------------------------------

class Snake:
    def __init__(self, game_board):
        self.game_board = game_board
        self.length = INITIAL_SNAKE_LENGTH
        self.body = self._find_initial_spawn()
        if not self.body:
            self.body = [
                (GRID_WIDTH // 2 - i, GRID_HEIGHT // 2) for i in range(INITIAL_SNAKE_LENGTH)
            ]
        self.direction = RIGHT
        self.grow_pending = 0

    # ---- initial spawn helpers ----------------------------------------

    def _find_initial_spawn(self):
        for r in range(GRID_HEIGHT // 2, GRID_HEIGHT):
            for c in range(GRID_WIDTH // 2, GRID_WIDTH):
                head = (c, r)
                if self._is_safe_spawn_area(head, INITIAL_SNAKE_LENGTH):
                    return [(head[0] - i, head[1]) for i in range(INITIAL_SNAKE_LENGTH)]
        return None

    def _is_safe_spawn_area(self, head_pos, length):
        for i in range(length):
            x, y = head_pos[0] - i, head_pos[1]
            if not self.game_board.is_valid_node((x, y)) or (
                (x, y) in self.game_board.obstacles or (x, y) in self.game_board.teleporters
            ):
                return False
        return True

    # -------------------------------------------------------------------

    def get_head_position(self):
        return self.body[0]

    def turn(self, direction):
        if self.length > 1 and (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        self.direction = direction

    def _compute_new_head(self, direction):
        new_head = (self.body[0][0] + direction[0], self.body[0][1] + direction[1])
        new_head = teleport_if_needed(new_head, self.game_board)
        return new_head

    def will_collide(self, direction):
        new_head = self._compute_new_head(direction)
        if not self.game_board.is_valid_node(new_head):
            return True
        if new_head in self.game_board.obstacles:
            return True
        if new_head in self.body[:-1]:  # allow tail because it moves away
            return True
        return False

    def move(self):
        new_head = self._compute_new_head(self.direction)

        # collision detection
        if (
            not self.game_board.is_valid_node(new_head)
            or new_head in self.game_board.obstacles
            or new_head in self.body[1:]
        ):
            return False  # crash

        self.body.insert(0, new_head)
        if self.grow_pending:
            self.grow_pending -= 1
            self.length += 1
        else:
            self.body.pop()
        return True

    def grow(self):
        self.grow_pending += 1

    def draw(self, surface):
        head_rect = pygame.Rect(
            self.body[0][0] * GRID_SIZE, self.body[0][1] * GRID_SIZE, GRID_SIZE, GRID_SIZE
        )
        pygame.draw.rect(surface, LIME_GREEN, head_rect)
        pygame.draw.rect(surface, BLACK, head_rect, 1)
        for seg in self.body[1:]:
            rect = pygame.Rect(seg[0] * GRID_SIZE, seg[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, GREEN, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class Food:
    def __init__(self, board, snake_body):
        self.board = board
        self.position = (0, 0)
        self.type = "normal"
        self.randomize_position(snake_body)

    def randomize_position(self, snake_body):
        available = [
            (c, r)
            for r in range(GRID_HEIGHT)
            for c in range(GRID_WIDTH)
            if (
                (c, r) not in snake_body
                and (c, r) not in self.board.obstacles
                and (c, r) not in self.board.teleporters
            )
        ]
        if not available:
            self.position = (-1, -1)
            return
        self.position = random.choice(available)
        self.type = "bonus" if random.random() < BONUS_FOOD_CHANCE else "normal"

    def draw(self, surface):
        if self.position == (-1, -1):
            return
        color = RED if self.type == "normal" else PINK
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)


class GameBoard:
    def __init__(self):
        self.obstacles = set()
        self.teleporters = {}
        self._create_level()

    def _create_level(self):
        for i in range(5, GRID_HEIGHT - 5):
            self.obstacles.add((GRID_WIDTH // 4, i))
            self.obstacles.add((GRID_WIDTH * 3 // 4, i))
        for i in range(5, GRID_WIDTH - 5):
            if i % 5 != 0:
                self.obstacles.add((i, GRID_HEIGHT // 3))
                self.obstacles.add((i, GRID_HEIGHT * 2 // 3))
        tp_pairs = [
            ((2, GRID_HEIGHT // 2), (GRID_WIDTH - 3, GRID_HEIGHT // 2)),
            ((GRID_WIDTH // 2, 2), (GRID_WIDTH // 2, GRID_HEIGHT - 3)),
        ]
        for entry, exit_ in tp_pairs:
            if entry not in self.obstacles and exit_ not in self.obstacles:
                self.teleporters[entry] = exit_
        # make sure exits not entries
        for entry, exit_ in list(self.teleporters.items()):
            if exit_ in self.teleporters or exit_ in self.obstacles:
                del self.teleporters[entry]

    def is_valid_node(self, pos):
        x, y = pos
        return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

    def draw(self, surface):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(surface, GRAY, (0, y), (SCREEN_WIDTH, y))
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(surface, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for obs in self.obstacles:
            rect = pygame.Rect(obs[0] * GRID_SIZE, obs[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, BLUE, rect)
        for entry, exit_ in self.teleporters.items():
            entry_rect = pygame.Rect(entry[0] * GRID_SIZE, entry[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            exit_rect = pygame.Rect(exit_[0] * GRID_SIZE, exit_[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, CYAN, entry_rect)
            pygame.draw.circle(surface, BLACK, entry_rect.center, GRID_SIZE // 4)
            pygame.draw.rect(surface, MAGENTA, exit_rect)
            pygame.draw.circle(surface, WHITE, exit_rect.center, GRID_SIZE // 4)


# --- AI / Autoplay -------------------------------------------------------

def bfs_shortest_path(start, goal, board, snake_body):
    if start == goal:
        return []
    # obstacles include snake body except tail
    obstacles = set(board.obstacles) | set(snake_body[:-1])
    frontier = deque([start])
    parent = {start: None}
    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for d in DIRS:
            nx, ny = current[0] + d[0], current[1] + d[1]
            neighbour = (nx, ny)
            neighbour = teleport_if_needed(neighbour, board)
            if not board.is_valid_node(neighbour) or neighbour in obstacles:
                continue
            if neighbour not in parent:
                parent[neighbour] = current
                frontier.append(neighbour)
    else:
        return []

    # reconstruct
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def find_safe_direction(snake, board):
    """Return a direction that doesn't crash next tick; None if none safe."""
    for d in DIRS:
        if not snake.will_collide(d):
            return d
    return None


# --- UI helpers ----------------------------------------------------------

def draw_text(surface, text, size, x, y, color=WHITE):
    font = pygame.font.Font(None, size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(midtop=(x, y))
    surface.blit(surf, rect)


def game_over_screen(screen, score):
    screen.fill(DARK_GRAY)
    draw_text(screen, "GAME OVER", 64, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    draw_text(screen, f"Score: {score}", 40, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    draw_text(screen, "Press R to Restart or Q to Quit", 30, SCREEN_WIDTH // 2, SCREEN_HEIGHT * 3 // 4)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    return


# --- Main loop -----------------------------------------------------------

def main_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake – Indestructible Autoplay")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    board = GameBoard()
    snake = Snake(board)
    food = Food(board, snake.body)
    score = 0

    autoplay = False
    path = []

    while True:
        click_pos = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    autoplay = not autoplay
                    path = []
                elif not autoplay:
                    if event.key == pygame.K_UP:
                        snake.turn(UP)
                    elif event.key == pygame.K_DOWN:
                        snake.turn(DOWN)
                    elif event.key == pygame.K_LEFT:
                        snake.turn(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        snake.turn(RIGHT)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click_pos = event.pos
        if click_pos and BUTTON_RECT.collidepoint(click_pos):
            autoplay = not autoplay
            path = []

        # ----------------------------------------------------------------
        if autoplay:
            # recompute if path empty or food changed
            if not path or path[-1] != food.position:
                path = bfs_shortest_path(snake.get_head_position(), food.position, board, snake.body)
            if not path:
                safe_dir = find_safe_direction(snake, board)
                if safe_dir:
                    snake.turn(safe_dir)
            else:
                nxt = path.pop(0)
                dx = nxt[0] - snake.get_head_position()[0]
                dy = nxt[1] - snake.get_head_position()[1]
                snake.turn((int(dx and dx / abs(dx)), int(dy and dy / abs(dy))))

        if not snake.move():
            # If autoplay is ON we retry safe direction to stay alive
            if autoplay:
                safe_dir = find_safe_direction(snake, board)
                if safe_dir:
                    snake.turn(safe_dir)
                    snake.move()
                else:
                    game_over_screen(screen, score); return
            else:
                game_over_screen(screen, score); return

        # food eaten
        if snake.get_head_position() == food.position:
            snake.grow()
            score += BONUS_FOOD_SCORE if food.type == "bonus" else NORMAL_FOOD_SCORE
            food.randomize_position(snake.body)
            path = []
            if food.position == (-1, -1):
                game_over_screen(screen, score); return

        # ---------------------------------------------------------------- drawing
        screen.fill(BLACK)
        board.draw(screen)
        snake.draw(screen)
        food.draw(screen)
        # highlight path
        if autoplay:
            for pos in path:
                rect = pygame.Rect(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, YELLOW, rect, 3)
        score_surf = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_surf, (10, 10))
        pygame.draw.rect(screen, (70, 70, 70), BUTTON_RECT, border_radius=5)
        btn_label = "AUTO ON" if autoplay else "AUTO OFF"
        txt = font.render(btn_label, True, WHITE)
        screen.blit(txt, txt.get_rect(center=BUTTON_RECT.center))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main_game()