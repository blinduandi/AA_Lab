import pygame
import random
import heapq
import sys
from collections import deque

# ----------------------------------------
# CONFIGURATION / CONSTANTS
# ----------------------------------------
CELL_SIZE = 40
MARGIN = 2
FONT_SIZE = 16

START_COLOR = (173, 216, 230)
GOAL_COLOR = (144, 238, 144)
PLAYER_COLORS = [(255, 0, 0), (0, 0, 255)]  # player 0 = red, player 1 = blue
MUD_COLOR = (139, 69, 19)
PATH_COLOR = (0, 255, 255)

TEXT_COLOR = (255, 255, 255)
BG_COLOR = (30, 30, 30)

MUD_WEIGHT_RANGE = (15, 25)
SHOW_PATH_EVENT = pygame.USEREVENT + 1
ADVANCE_EVENT = pygame.USEREVENT + 2

# ----------------------------------------
# LEVEL & MAZE LOGIC
# ----------------------------------------
class Level:
    DIAGONAL_DIRS = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, level_num: int):
        self.level_num = level_num
        self.size = 6 + level_num
        self.weights = [[random.randint(1, 9) for _ in range(self.size)]
                        for _ in range(self.size)]
        # mud tiles
        mud_positions = [(x, y) for x in range(self.size) for y in range(self.size)
                         if (x, y) not in [(0, 0), (self.size - 1, self.size - 1)]]
        for x, y in random.sample(mud_positions, k=min(level_num, (self.size*self.size)//5)):
            self.weights[x][y] = random.randint(*MUD_WEIGHT_RANGE)
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
# DUAL GAME FOR 1v1
# ----------------------------------------
class DualGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        w, h = self.screen.get_size()
        self.pane_w = w // 2
        self.pane_h = h
        self.font = pygame.font.SysFont('consolas', FONT_SIZE)
        # states per player
        self.states = [self._new_state(), self._new_state()]
        self.clock = pygame.time.Clock()
        self.active = 0  # last mover

    def _new_state(self):
        lvl = Level(1)
        return {
            'level': lvl,
            'player': (0, 0),
            'cost': lvl.weights[0][0],
            'finished': False,
            'hint': False,
            'path_idx': 0,
            'wins': 0,
            'losses': 0,
        }

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    self._handle_key(ev.key)
            self._draw()
            self.clock.tick(60)

    def _handle_key(self, key):
        # mapping: (player_idx, dy, dx) or commands
        map_list = [
            {pygame.K_LEFT:(0,0,-1), pygame.K_RIGHT:(0,0,1), pygame.K_UP:(0,-1,0), pygame.K_DOWN:(0,1,0),
             pygame.K_q:(0,-1,-1), pygame.K_e:(0,-1,1), pygame.K_z:(0,1,-1), pygame.K_c:(0,1,1),
             pygame.K_h:('hint',0), pygame.K_t:('cheat',0)},
            {pygame.K_a:(1,0,-1), pygame.K_d:(1,0,1), pygame.K_w:(1,-1,0), pygame.K_s:(1,1,0),
             pygame.K_y:(1,-1,-1), pygame.K_u:(1,-1,1), pygame.K_h:(1,1,-1), pygame.K_j:(1,1,1),
             pygame.K_k:('hint',1), pygame.K_l:('cheat',1)}
        ]
        for m in map_list:
            if key in m:
                cmd = m[key]
                break
        else:
            return
        if isinstance(cmd[0], str):
            act, idx = cmd
            st = self.states[idx]
            if act == 'hint':
                st['hint'] = not st['hint']
            else:  # cheat
                st['path_idx'] = len(st['level'].opt_path)
            return
        # movement
        idx, dy, dx = cmd
        st = self.states[idx]
        self.active = idx
        if st['finished']:
            return
        x,y = st['player']
        nx, ny = x+dx, y+dy
        lvl = st['level']
        if 0 <= nx < lvl.size and 0 <= ny < lvl.size:
            st['player'] = (nx, ny)
            st['cost'] += lvl.weights[nx][ny]
            if (nx, ny) == (lvl.size-1, lvl.size-1):
                st['finished'] = True
                # Determine winner and reset opponent
                # Increment win for this player
                st['wins'] += 1
                # Advance this player's level
                new_level = st['level'].level_num + 1
                st['level'] = Level(new_level)
                st['player'] = (0, 0)
                st['cost'] = st['level'].weights[0][0]
                st['finished'] = False
                st['hint'] = False
                st['path_idx'] = 0
                # Reset opponent
                opp = self.states[1-idx]
                opp['losses'] += 1
                opp['level'] = Level(1)
                opp['player'] = (0, 0)
                opp['cost'] = opp['level'].weights[0][0]
                opp['finished'] = False
                opp['hint'] = False
                opp['path_idx'] = 0

    def _draw(self):
        self.screen.fill(BG_COLOR)
        # draw legend
        legend = [
            "Left: Arrows/QEZC, H=hint, T=cheat",
            "Right: WASD/YUHJ, K=hint, L=cheat, ESC=quit"
        ]
        for i, txt in enumerate(legend):
            surf = self.font.render(txt, True, TEXT_COLOR)
            x = (self.screen.get_width()-surf.get_width())//2
            self.screen.blit(surf, (x, 10 + i*(FONT_SIZE+4)))
        # draw each pane
        for idx, st in enumerate(self.states):
            xoff = idx * self.pane_w
            # highlight active
            if idx == self.active:
                pygame.draw.rect(self.screen, PLAYER_COLORS[idx], (xoff,0,self.pane_w,self.pane_h), 4)
            lvl = st['level']; size = lvl.size
            # draw grid
            for i in range(size):
                for j in range(size):
                    rect = pygame.Rect(xoff + MARGIN + (CELL_SIZE+MARGIN)*i,
                                       MARGIN + (CELL_SIZE+MARGIN)*j,
                                       CELL_SIZE, CELL_SIZE)
                    color = MUD_COLOR if lvl.weights[i][j]>=MUD_WEIGHT_RANGE[0] else (200,200,200)
                    pygame.draw.rect(self.screen, color, rect)
                    if (i,j)==(0,0): pygame.draw.rect(self.screen, START_COLOR, rect)
                    if (i,j)==(size-1,size-1): pygame.draw.rect(self.screen, GOAL_COLOR, rect)
                    # path
                    if st['finished'] or st['hint']:
                        if (i,j) in lvl.opt_path[:st['path_idx'] if st['finished'] else len(lvl.opt_path)]:
                            pygame.draw.rect(self.screen, PATH_COLOR, rect,4)
                    # weight
                    surf = self.font.render(str(lvl.weights[i][j]), True, TEXT_COLOR)
                    self.screen.blit(surf, (rect.x+CELL_SIZE//2-surf.get_width()//2,
                                             rect.y+CELL_SIZE//2-surf.get_height()//2))
            # player
            px,py = st['player']
            prect = pygame.Rect(xoff + MARGIN + (CELL_SIZE+MARGIN)*px,
                                 MARGIN + (CELL_SIZE+MARGIN)*py,
                                 CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, PLAYER_COLORS[idx], prect, 3)
            # HUD
            y0 = size*(CELL_SIZE+MARGIN) + 20
            lines = [f"L{st['level'].level_num} C:{st['cost']}/{st['level'].opt_cost}",
                     f"W:{st['wins']} L:{st['losses']}"]
            if st['hint']: lines.append('Hint')
            for k, line in enumerate(lines):
                surf = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(surf, (xoff+10, y0 + k*(FONT_SIZE+2)))
        pygame.display.flip()

if __name__ == '__main__':
    DualGame().run()
