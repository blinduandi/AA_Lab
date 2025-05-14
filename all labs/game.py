import pygame
import random
import heapq
import sys
from collections import deque

# ----------------------------------------
# CONFIGURATION / CONSTANTS
# ----------------------------------------
CELL_SIZE = 40               # pixels per tile
MARGIN = 2                   # gap between tiles
FONT_SIZE = 16

START_COLOR = (173, 216, 230)
GOAL_COLOR = (144, 238, 144)
PLAYER_COLOR = (255, 0, 0)
MUD_COLOR = (139, 69, 19)     # visually distinct "slow mud" tiles
PATH_COLOR = (0, 255, 255)    # overlay for cheapest-path animation

TEXT_COLOR = (0, 0, 0)
BG_COLOR = (30, 30, 30)

MUD_WEIGHT_RANGE = (15, 25)   # cost range for mud tiles

# Custom user events
SHOW_PATH_EVENT = pygame.USEREVENT + 1      # reveals next cell of cheapest path
ADVANCE_LEVEL_EVENT = pygame.USEREVENT + 2  # loads next level after optimal finish

# Medal thresholds (cost_ratio = player / optimal)
GOLD_THRESHOLD = 1.00     # must equal optimal to level-up
SILVER_THRESHOLD = 1.05   # ≤ 5 % over optimal
BRONZE_THRESHOLD = 1.20   # ≤ 20 % over optimal

# ----------------------------------------
# LEVEL LOGIC (grid + Dijkstra)
# ----------------------------------------
class Level:
    """Single maze with weighted tiles & cheapest-path pre-computation."""

    DIAGONAL_DIRS = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, level_num: int):
        self.level_num = level_num
        self.size = 6 + level_num               # grid grows each level

        # --- Generate weights ---
        self.weights = [
            [random.randint(1, 9) for _ in range(self.size)]
            for _ in range(self.size)
        ]

        # --- Sprinkle "mud" tiles ---
        mud_tiles = random.sample(
            [(x, y) for x in range(self.size) for y in range(self.size)
             if (x, y) not in [(0, 0), (self.size - 1, self.size - 1)]],
            k=min(level_num, (self.size * self.size) // 5)
        )
        for x, y in mud_tiles:
            self.weights[x][y] = random.randint(*MUD_WEIGHT_RANGE)

        # Pre-compute optimal path & cost
        self.opt_cost, self.opt_path = self._dijkstra()

    def _neighbors(self, x, y):
        for dx, dy in self.ORTHO_DIRS + self.DIAGONAL_DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                yield nx, ny

    def _dijkstra(self):
        start, goal = (0, 0), (self.size - 1, self.size - 1)
        dist = [[float('inf')] * self.size for _ in range(self.size)]
        dist[0][0] = self.weights[0][0]
        prev = {}
        pq = [(dist[0][0], start)]

        while pq:
            d, (x, y) = heapq.heappop(pq)
            if (x, y) == goal:
                break
            if d != dist[x][y]:
                continue
            for nx, ny in self._neighbors(x, y):
                nd = d + self.weights[nx][ny]
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nd, (nx, ny)))

        path = deque()
        node = goal
        while node != start:
            path.appendleft(node)
            node = prev.get(node, start)
        path.appendleft(start)
        return dist[goal[0]][goal[1]], list(path)

# ----------------------------------------
# GAME LOOP / STATE MACHINE
# ----------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.SysFont('consolas', FONT_SIZE)

        self.total_wins = 0
        self.total_losses = 0
        self.level_num = 1
        self.show_hint = False              # toggle for showing full path anytime
        self._load_level()

    def _load_level(self):
        self.level = Level(self.level_num)
        self.player = (0, 0)
        self.cost = self.level.weights[0][0]
        self.finished = False
        self.message = ''
        self.medal = None
        self._path_reveal_index = 0

        side = self.level.size * CELL_SIZE + (self.level.size + 1) * MARGIN
        self.screen = pygame.display.set_mode((side, side + 100))
        pygame.display.set_caption(f'Cheapest Path Trainer – Level {self.level_num}')

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    # Toggle hint mode with 'H'
                    if event.key == pygame.K_h:
                        self.show_hint = not self.show_hint
                    # Cheat key: reveal full optimal path anytime
                    elif event.key == pygame.K_t:
                        self._path_reveal_index = len(self.level.opt_path)
                        pygame.time.set_timer(SHOW_PATH_EVENT, 0)
                    else:
                        self._handle_move(event.key)

                if event.type == SHOW_PATH_EVENT:
                    self._path_reveal_index += 1
                    if self._path_reveal_index >= len(self.level.opt_path):
                        pygame.time.set_timer(SHOW_PATH_EVENT, 0)

                if event.type == ADVANCE_LEVEL_EVENT:
                    pygame.time.set_timer(ADVANCE_LEVEL_EVENT, 0)
                    if self.medal == 'Gold':
                        self.level_num += 1
                    self._load_level()

            self._draw()
            clock.tick(60)

    def _handle_move(self, key):
        dx = dy = 0
        if key == pygame.K_LEFT:       dx = -1
        elif key == pygame.K_RIGHT:    dx = 1
        elif key == pygame.K_UP:       dy = -1
        elif key == pygame.K_DOWN:     dy = 1
        elif key == pygame.K_q:        dx = dy = -1
        elif key == pygame.K_e:        dx, dy = 1, -1
        elif key == pygame.K_z:        dx, dy = -1, 1
        elif key == pygame.K_c:        dx = dy = 1
        else:
            return

        nx, ny = self.player[0] + dx, self.player[1] + dy
        if 0 <= nx < self.level.size and 0 <= ny < self.level.size:
            self.player = (nx, ny)
            self.cost += self.level.weights[nx][ny]
            if (nx, ny) == (self.level.size - 1, self.level.size - 1):
                self._complete_level()

    def _complete_level(self):
        self.finished = True
        ratio = self.cost / self.level.opt_cost
        if abs(ratio - 1) < 1e-9:
            self.medal = 'Gold'
            self.message = 'Optimal! Gold medal – advancing…'
            self.total_wins += 1
            pygame.time.set_timer(ADVANCE_LEVEL_EVENT, 2000)
        elif ratio <= SILVER_THRESHOLD:
            self.medal = 'Silver'
            self.message = f"{self.medal}! Cost {self.cost} (opt {self.level.opt_cost}) – try again!"
            self.total_losses += 1
            pygame.time.set_timer(ADVANCE_LEVEL_EVENT, 3000)
        elif ratio <= BRONZE_THRESHOLD:
            self.medal = 'Bronze'
            self.message = f"{self.medal}. Cost {self.cost} (opt {self.level.opt_cost}) – try again!"
            self.total_losses += 1
            pygame.time.set_timer(ADVANCE_LEVEL_EVENT, 3000)
        else:
            self.medal = None
            self.message = f"Over budget: {self.cost} vs {self.level.opt_cost}. Try again!"
            self.total_losses += 1
            pygame.time.set_timer(ADVANCE_LEVEL_EVENT, 3000)

        # Start cheapest-path animation
        self._path_reveal_index = 0
        pygame.time.set_timer(SHOW_PATH_EVENT, 120)

    def _draw(self):
        self.screen.fill(BG_COLOR)
        size = self.level.size
        for x in range(size):
            for y in range(size):
                rect = pygame.Rect(
                    MARGIN + (CELL_SIZE + MARGIN) * x,
                    MARGIN + (CELL_SIZE + MARGIN) * y,
                    CELL_SIZE,
                    CELL_SIZE
                )
                # Base tile color
                is_mud = self.level.weights[x][y] >= MUD_WEIGHT_RANGE[0]
                base = MUD_COLOR if is_mud else (200, 200, 200)
                pygame.draw.rect(self.screen, base, rect)
                if (x, y) == (0, 0): pygame.draw.rect(self.screen, START_COLOR, rect)
                if (x, y) == (size-1, size-1): pygame.draw.rect(self.screen, GOAL_COLOR, rect)

                                # Overlay cheapest-path
                if self.finished:
                    # animate slice
                    if (x, y) in self.level.opt_path[:self._path_reveal_index]:
                        pygame.draw.rect(self.screen, PATH_COLOR, rect, 4)
                elif self.show_hint:
                    # show full optimal path
                    if (x, y) in self.level.opt_path:
                        pygame.draw.rect(self.screen, PATH_COLOR, rect, 4)

                # Weight label
                txt = self.font.render(str(self.level.weights[x][y]), True, TEXT_COLOR)
                self.screen.blit(txt, (rect.x + CELL_SIZE//2 - txt.get_width()//2,
                                       rect.y + CELL_SIZE//2 - txt.get_height()//2))

        # Player indicator
        px, py = self.player
        prect = pygame.Rect(
            MARGIN + (CELL_SIZE + MARGIN) * px,
            MARGIN + (CELL_SIZE + MARGIN) * py,
            CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(self.screen, PLAYER_COLOR, prect, 3)

        # HUD
        hud_top = size * CELL_SIZE + (size + 1) * MARGIN
        lines = [f"Level {self.level_num}  Cost: {self.cost} (opt {self.level.opt_cost})",
                 f"Wins: {self.total_wins}  Losses: {self.total_losses}"]
        if self.message: lines.append(self.message)
        if self.show_hint: lines.append("Hint: full path shown")
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (255,255,255))
            self.screen.blit(surf, (10, hud_top + 10 + i*22))

        pygame.display.flip()

if __name__ == '__main__':
    Game().run()
