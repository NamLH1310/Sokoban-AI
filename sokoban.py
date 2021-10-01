from queue import Queue, PriorityQueue, LifoQueue
from time import time
from math import sqrt
import sys
import tracemalloc

exec_times = []
memory_profile = []

class State:
    def __init__(self,
                walls: set,
                boxes: set,
                storages: set,
                player: (int, int),
                path: str,
                width: int,
                height: int,
                dead_locks: set,
                parent,
                euclidean_hrs=True):
        self.walls = walls
        self.boxes = boxes
        self.storages = storages
        self.player = player
        self.path = path
        self.width = width
        self.height = height
        self.dead_locks = dead_locks
        self.parent = parent
        self.euclidean_hrs = euclidean_hrs
        self.successors = []


    def is_in_bound(self, x: int, y: int):
        return 0 <= x < self.height and 0 <= y < self.width


    def move(self,
            ax: int,
            ay: int,
            bx: int,
            by: int,
            d: str):
        if not self.is_in_bound(ax, ay) or not self.is_in_bound(bx, by):
            return
        
        changed = False
        attempt = ax, ay
        new_box = bx, by

        if attempt not in self.walls:
            if attempt not in self.boxes or new_box not in self.boxes and new_box not in self.walls:
                if attempt in self.boxes:
                    if new_box in self.dead_locks:
                        return
                    self.boxes.remove(attempt)
                    self.boxes.add(new_box)
                    changed = True

                new_state = State(self.walls,
                               self.boxes.copy(),
                               self.storages.copy(),
                               attempt,
                               self.path + d,
                               self.width,
                               self.height,
                               self.dead_locks,
                               self,
                               self.euclidean_hrs)

                self.successors.append(new_state)

                if changed:
                    self.boxes.remove(new_box)
                    self.boxes.add(attempt)


    def get_successors(self):
        x, y = self.player

        # move up
        self.move(x - 1, y, x - 2, y, 'u')
        # move down
        self.move(x + 1, y, x + 2, y, 'd')
        # move left
        self.move(x, y - 1, x, y - 2, 'l')
        # move right
        self.move(x, y + 1, x, y + 2, 'r')
        
        return self.successors
    
    
    def has_reached_goal(self):
        for box in self.boxes:
            if box not in self.storages:
                return False

        return True

    
    def __eq__(self, obj):
        if type(self) is type(obj):
            return self.boxes == obj.boxes and self.player == obj.player
        return False


    def __ne__(self, obj):
        return not self == obj

    
    def __hash__(self):
        hash = 0
        for x, y in self.boxes:
            hash += x*31 + y
            hash *= 37
        return hash + self.player[0]*73 + self.player[1]


    def load_map(self):
        # Draw the map
        map = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        for i, j in self.walls:
            map[i][j] = '#'

        for i, j in self.storages:
            map[i][j] = '.'
        
        for i, j in self.boxes:
            if map[i][j] == '.':
                map[i][j] = '*'
            else:
                map[i][j] = '$'

        i, j = self.player

        if map[i][j] == '.':
            map[i][j] = '!'
        else:
            map[i][j] = '@'

        res = ''
        for row in map:
            res += ''.join(row)
            res += '\n'
        
        return res, map
    

    def __str__(self):
        drawed_map, _ = self.load_map()
        return drawed_map


    def print_path(self):
        path = []
        current_state = self
        while current_state is not None:
            path.append(str(current_state))
            current_state = current_state.parent

        walk = 0
        for map in path[::-1]:
            print(map)
            if walk < len(self.path):
                s = ''
                if self.path[walk] == 'u':
                    s = 'up'
                elif self.path[walk] == 'd':
                    s = 'down'
                elif self.path[walk] == 'r':
                    s = 'right'
                elif self.path[walk] == 'l':
                    s = 'left'
                print('go', s)
            walk += 1


    def euclidean(self):
        x, y = self.player

        player_to_boxes = 0

        for box in self.boxes:
            bx, by = box
            player_to_boxes += sqrt((x - bx)**2 + (y - by)**2)

        boxes_to_storage = 0

        for storage in self.storages:
            sx, sy = storage
            for box in self.boxes:
                bx, by = box
                boxes_to_storage += sqrt((bx - sx)**2 + (by - sy)**2)
        
        return player_to_boxes + boxes_to_storage

    def manhatten(self):
        x, y = self.player

        player_to_boxes = 0

        for box in self.boxes:
            bx, by = box
            player_to_boxes += abs(x - bx) + abs(y - by)

        boxes_to_storage = 0

        for storage in self.storages:
            sx, sy = storage
            for box in self.boxes:
                bx, by = box
                boxes_to_storage += abs(bx - sx) + abs(by - sy)

        return player_to_boxes + boxes_to_storage


    def __lt__(self, obj):
        if type(self) is type(obj):
            if self.euclidean_hrs:
                return self.euclidean() + len(self.path) > obj.euclidean() + len(obj.path)
            return self.manhatten() + len(self.path) > obj.manhatten() + len(obj.path)
        raise TypeError('Wrong type for comparison')

    
class Sokoban:
    '''
    An instance of sokoban game
    '''
    def __init__(self,
                map: list[str],
                boxes: set,
                storages: set,
                player: (int, int)):
        self.walls, self.width, self.height = self.get_map_info(map)
        self.boxes = boxes
        self.storages = storages
        self.player = player

    def get_map_info(self, map: list[str]):
        '''
        Return sokoban map information: coordinate of the wall, width, height
        '''
        walls = set()
        width = len(map[0])
        height = len(map)
        
        for i, row in enumerate(map):
            for j, ch in enumerate(row):
                if ch == '#':
                    walls.add((i, j))
        
        return walls, width, height

    def load_map(self):
        # Draw the map
        map = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        i, j = self.player

        map[i][j] = '@'

        for i, j in self.walls:
            map[i][j] = '#'
        
        for i, j in self.storages:
            map[i][j] = '.'

        for i, j in self.boxes:
            if map[i][j] == '.':
                map[i][j] = '*'
            else:
                map[i][j] = '$'


        drawed_map = ''
        for row in map:
            drawed_map += ''.join(row)
            drawed_map += '\n'
        
        return drawed_map, map
    

    def __str__(self):
        drawed_map, _ = self.load_map()
        return drawed_map


class DeadLockDetector:
    def __init__(self, soko: Sokoban):
        self.walls = soko.walls
        self.storages = soko.storages
        self.width = soko.width
        self.height = soko.height
        _, self.map = soko.load_map()
        self.dead_locks = self.find_dead_locks()

    
    def find_dead_locks(self):
        dead_locks = set()
        for i, row in enumerate(self.map):
            for j, _ in enumerate(row):
                current = i, j
                if current not in self.walls and current not in self.storages:
                    if self.is_corner(current) or self.is_boundary(current):
                        dead_locks.add(current)
        
        return dead_locks


    def is_corner(self, current: (int, int)) -> bool:
        x, y = current

        up = x - 1, y
        down = x + 1, y
        left = x, y - 1
        right = x, y + 1

        return up in self.walls and right in self.walls\
            or right in self.walls and down in self.walls\
            or down in self.walls and left in self.walls\
            or left in self.walls and up in self.walls

    
    def is_boundary(self, current: (int, int)) -> bool:
        x, y = current

        up = x - 1, y
        down = x + 1, y
        left = x, y - 1
        right = x, y + 1

        if left in self.walls:
            upbound = self.find_nearest_upbound(current)
            downbound = self.find_nearest_downbound(current)

            if upbound == -1 or downbound == -1:
                return False
            else:
                for m in range(upbound + 1, downbound):
                    if (m, y - 1) not in self.walls:
                        return False
            return True
        
        if right in self.walls:
            upbound = self.find_nearest_upbound(current)
            downbound = self.find_nearest_downbound(current)

            if upbound == -1 or downbound == -1:
                return False
            else:
                for m in range(upbound + 1, downbound):
                    if (m, y + 1) not in self.walls:
                        return False
            return True


        if up in self.walls:
            leftbound = self.find_nearest_leftbound(current)
            rightbound = self.find_nearest_rightbound(current)

            if rightbound == -1 or leftbound == -1:
                return False
            else:
                for m in range(leftbound + 1, rightbound):
                    if (x - 1, m) not in self.walls:
                        return False
            return True

        if down in self.walls:
            leftbound = self.find_nearest_leftbound(current)
            rightbound = self.find_nearest_rightbound(current)

            if rightbound == -1 or leftbound == -1:
                return False
            else:
                for m in range(leftbound + 1, rightbound):
                    if (x + 1, m) not in self.walls:
                        return False
            return True
        
        return False


    def find_nearest_upbound(self, current) -> int:
        x, y = current
        x -= 1

        while x >= 0:
            if (x, y) in self.walls:
                return x
            elif (x, y) in self.storages:
                return -1
            x -= 1

        return 0

    
    def find_nearest_downbound(self, current) -> int:
        x, y = current
        x += 1

        while x < self.height:
            if (x, y) in self.walls:
                return x
            elif (x, y) in self.storages:
                return -1
            x += 1

        return self.height


    def find_nearest_rightbound(self, current) -> int:
        x, y =  current
        y += 1

        while y < self.width:
            if (x, y) in self.walls:
                return y
            elif (x, y) in self.storages:
                return -1
            y += 1

        return self.width


    def find_nearest_leftbound(self, current) -> int:
        x, y = current
        y -= 1

        while y >= 0:
            if (x, y) in self.walls:
                return y
            elif (x, y) in self.storages:
                return -1
            y -= 1

        return 0

def dfs(init_state):
    s = LifoQueue()
    visited_state = set()

    s.put(init_state)
    visited_state.add(init_state)

    while not s.empty():
        state = s.get()
        if state.has_reached_goal():
            end_time = time()
            state.print_path()
            return True, end_time
        for neighbor in state.get_successors():
            if neighbor not in visited_state:
                s.put(neighbor)
                visited_state.add(neighbor)

    return False, None

def bfs(init_state):
    q = Queue()
    visited_state = set()

    q.put(init_state)
    visited_state.add(init_state)

    while not q.empty():
        state = q.get()
        if state.has_reached_goal():
            end_time = time()
            state.print_path()
            return True, end_time
        for neighbor in state.get_successors():
            if neighbor not in visited_state:
                q.put(neighbor)
                visited_state.add(neighbor)

    return False, None

def astar(init_state):
    q = PriorityQueue()
    visited_state = set()

    q.put(init_state)
    visited_state.add(init_state)

    while not q.empty():
        state = q.get()
        if state.has_reached_goal():
            end_time = time()
            state.print_path()
            return True, end_time
        for neighbor in state.get_successors():
            if neighbor not in visited_state:
                q.put(neighbor)
                visited_state.add(neighbor)

    return False, None
    

def main():
    sokobans = [
            Sokoban([
                    '########',
                    '###   ##',
                    '#   # ##',
                    '# #    #',
                    '#    # #',
                    '## #   #',
                    '##   ###',
                    '########',
                    ],
                    {(2, 2), (5, 2)},
                    {(3, 5), (5, 4)},
                    (6, 2)
                    ),
            Sokoban([
                    '########',
                    '###   ##',
                    '#   # ##',
                    '# #    #',
                    '#    # #',
                    '## #   #',
                    '##   ###',
                    '########',
                    ],
                    {(2, 2), (5, 2), (5, 5)},
                    {(4, 2), (5, 4), (3, 5)},
                    (6, 2)
                    ),
            Sokoban([
                    '#######',
                    '###  ##',
                    '##   ##',
                    '#     #',
                    '#   # #',
                    '# #   #',
                    '#   ###',
                    '#######',
                    ],
                    {(3, 2), (3, 3), (3, 4)},
                    {(2, 3), (4, 1), (4, 3)},
                    (3, 1)
                    ),
            Sokoban([
                    '#########',
                    '#  ###  #',
                    '#       #',
                    '#       #',
                    '###   ###',
                    '###   ###',
                    '#########',
                    ],
                    {(2, 2), (2, 4), (2, 6), (4, 5)},
                    {(2, 4), (3, 4), (4, 4), (5, 4)},
                    (3, 4)
                    ),
            Sokoban([
                    '#######',
                    '##   ##',
                    '## # ##',
                    '##    #',
                    '#  #  #',
                    '#    ##',
                    '#  # ##',
                    '##   ##',
                    '#######',
                    ],
                    {(2, 4), (4, 4), (6, 4)},
                    {(4, 2), (5, 2), (6, 2)},
                    (3, 3)
                    ),
            Sokoban([
                    '##########',
                    '######  ##',
                    '###     ##',
                    '###   # ##',
                    '#    #   #',
                    '#        #',
                    '####  ####',
                    '##########',
                    ],
                    {(3, 4), (4, 3), (5, 2), (5, 6)},
                    {(2, 3), (2, 4), (4, 6), (4, 7)},
                    (3, 5)
                    ),
            Sokoban([
                    '##########',
                    '######   #',
                    '##     # #',
                    '## #     #',
                    '## ## # ##',
                    '#       ##',
                    '# ##   ###',
                    '#     ####',
                    '##########',
                    ],
                    {(2, 3), (3, 5), (3, 6), (5, 3), (5, 5)},
                    {(2, 5), (5, 4), (5, 6), (7, 3), (7, 5)},
                    (7, 4)
                    ),
            Sokoban([
                    '#########',
                    '###   ###',
                    '#   #  ##',
                    '#      ##',
                    '## # #  #',
                    '##      #',
                    '###  ####',
                    '#########',
                    ],
                    {(2, 3), (3, 6), (5, 3)},
                    {(3, 3), (3, 5), (4, 2)},
                    (3, 4)
                    ),
            ]

    search = None
    filename = ''
    while search_strategy := input('Type 1 to use DFS, 2 to use A star: '):
        if search_strategy == '1':
            search = dfs
            filename = 'time_benchmark.txt'
            break
        if search_strategy == '2':
            search = astar
            filename = 'time_benchmark_astar.txt'
            break
    
    for sokoban in sokobans:
        dead_locks = DeadLockDetector(sokoban).dead_locks
        init_state = State(sokoban.walls,
                            sokoban.boxes,
                            sokoban.storages,
                            sokoban.player,
                            '',
                            sokoban.width,
                            sokoban.height,
                            dead_locks,
                            None,
                            False)

        # trace memory
        ## tracemalloc.start()

        # trace time
        start_time = time()
        reached_goal, end_time = search(init_state)

        ## memory_con = round(tracemalloc.get_traced_memory()[0]/1024**2, 4)

        ## tracemalloc.stop()

        if reached_goal:
        # The memory tracing calcution roughly double the executional time of each test case
            exec_time = round(end_time - start_time, 4)
            exec_times.append(exec_time)
            print(f'Success\nSolved in {exec_time}s')
        else:
            print(f'Cannot find solution')

        print('----------------------------------')

        ## memory_profile.append(memory_con)
        ## print(f'Consumed {memory_con}mb')


    out = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        print(exec_times)
        sys.stdout = out

if __name__ == '__main__':
    main()
