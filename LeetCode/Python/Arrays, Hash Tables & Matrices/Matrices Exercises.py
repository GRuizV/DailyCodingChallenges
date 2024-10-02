'''
CHALLENGES INDEX

54. Spiral Matrix (Matrix)  
79. Word Search (Matrix) (BT)
130. Surrounded Regions (Matrix) (BFS) (DFS)
200. Number of Islands (Matrix) (DFS)
212. Word Search II (Array) (DFS) (BT) (Matrix)
240. Search a 2D Matrix II (Matrix) (DQ) (BS)
289. Game of Life (Matrix)

51. N-Queens (Matrix) (BT)
64. Minimum Path Sum (Matrix) (DP)
74. Search a 2D Matrix (BS) (Matrix)


*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others


(11)
'''


'54. Spiral Matrix'
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,2,3],[4,5,6],[7,8,9]]
#     # Output: [1,2,3,6,9,8,7,4,5]

#     # Case 2
#     matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
#     # Output: [1,2,3,4,8,12,11,10,9,5,6,7]

#     # Custom Case  / m = 1
#     matrix = [[2,1,3]]
#     # Output: [2,1,3]

#     # Custom Case / n = 1
#     matrix = [[2],[1],[3],[5],[4]]
#     # Output: [2,1,3,5,4]

#     # Custom Case
#     matrix = [[1,2],[3,4]]
#     # Output: [1,2,4,3]

#     # Custom Case
#     matrix = [[2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16]]
#     # Output: [2,3,4,7,10,13,16,15,14,11,8,5,6,9,12]

#     '''
#     My approach

#         Intuition:
#             1. Handle Special Cases: m = 1 / n = 1 

#             2. Make a while loop that runs until the input has no elements:
#                 a. Take all the subelements from the first element and append them individually to the result.
#                 b. Make a for loop and take the last subelement of each element and append them individually to the result
#                 c. Take all the subelements from the last element and append them in a reverse order individually to the result.
#                 d. Make a for loop and take the first subelement of each element and append them in a reverse order individually to the result.

#             3. Return the result
#     '''

#     def spiralOrder(matrix):

#         if len(matrix) == 1:
#             return matrix[0]
        
#         if len(matrix[0]) == 1:
#             return [num for vec in matrix for num in vec]
        

#         result = []

#         while len(matrix) != 0:

#             first_element = matrix.pop(0)
#             result += first_element

#             if len(matrix) == 0:
#                 break

#             second_element = []
#             for elem in matrix:
#                 second_element.append(elem.pop(-1))
#             result += second_element

#             third_element = matrix.pop(-1)
#             result += reversed(third_element)
            
#             if len(matrix) > 0 and len(matrix[0]) == 0:
#                 break

#             fourth_element = []
#             for elem in matrix:
#                 fourth_element.append(elem.pop(0))
#             result += reversed(fourth_element)

#         return result

#     print(spiralOrder(matrix))


#     'Notes: it works up to 76% of the cases, but from here seems more like patching something that could be better designed'


#     'Another Approach'

#     def spiralOrder(matrix):
            
#             result = []

#             while matrix:
#                 result += matrix.pop(0) # 1

#                 if matrix and matrix[0]: # 2 
#                     for line in matrix:
#                         result.append(line.pop())

#                 if matrix: # 3
#                     result += matrix.pop()[::-1]

#                 if matrix and matrix[0]: # 4

#                     for line in matrix[::-1]:
#                         result.append(line.pop(0))

#             return result

#     'Notes: Same logic, better executed'

'79. Word Search'
# def x():

#     # Input
#     # Case 1
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'ABCCED'
#     # Output: True

#     # Case 2
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'SEE'
#     # Output: True

#     # Case 3
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'ABCB'
#     # Output: False

#     '''
#     Intuition:

#         The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
#         At each cell, we check if the current character matches the corresponding character of the word. 
#         If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

#         Approach

#             1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
#             2. Base cases:
#                 - If k equals the length of the word, return True, indicating that the word has been found.
#                 - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
#             3. Mark the current cell as visited by changing its value or marking it as empty.
#             4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
#             5. If any recursive call returns True, indicating that the word has been found, return True.
#             6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
#             7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
            
#     '''

#     'Backtracking (Recursive) Approach'
#     def exist(board: list[list[str]], word: str) -> bool:

#         def backtrack(i, j, k):

#             if k == len(word):
#                 return True
            
#             if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = ''
            
#             if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#                 return True
            
#             board[i][j] = temp

#             return False

#         for i in range(len(board)):

#             for j in range(len(board[0])):

#                 if backtrack(i, j, 0):

#                     return True

#         return False
            
#     # Testing
#     print(exist(board, word))

'''130. Surrounded Regions'''
# def x():

#     #Input
#     #Case 1
#     board = [
#         ["X","X","X","X"],
#         ["X","O","O","X"],
#         ["X","X","O","X"],
#         ["X","O","X","X"]
#         ]
#     # output = [
#     #     ["X","X","X","X"],
#     #     ["X","X","X","X"],
#     #     ["X","X","X","X"],
#     #     ["X","O","X","X"]
#     #     ]

#     #Case 2
#     board = [
#         ['X']
#         ]
#     # output = [
#         # ['X']
#         # ]

#     #Custom Case
#     board = [["O","O"],["O","O"]]


#     '''
#     My Approach

#         Intuition:
#             1. Check if there is any 'O' at the boarders.
#             2. Check is there is any 'O' adjacent to the one in the boarder:
#                 - If do, add them to the not-be-flipped ground and re run.
#                 - if doesn't, flip everything to 'X' and return
#             (Do this until there is no 'O' unchecked )
#     '''

#     def solve(board:list[list[str]]) -> None:

#         M = len(board)
#         N = len(board[0])

#         no_flip = []
#         all_os = []


#         # Collect all 'O's
#         for i in range(M):
#             all_os.extend((i,j) for j in range(N) if board[i][j] == 'O')
        

#         #   Check if there is a boarder 'O' within the group
#         for i in range(len(all_os)):

#             if all_os[i][0] in (0, M-1) or all_os[i][1] in (0, N-1):
#                 no_flip.append(all_os[i])


#         # Collect the 'O's near to no_flip 'O' iteratively
#         flipped = None
#         i = 0

#         while True:

#             # Condition to end the loop
#             if len(all_os) == 0 or i == len(all_os) and flipped is False:
#                 break

#             #Collecting the possibilities of an adjacent 'O'
#             adjacents = []

#             for pos in no_flip:
#                 adjacents.extend([(pos[0]-1, pos[1]), (pos[0]+1, pos[1]), (pos[0], pos[1]-1), (pos[0], pos[1]+1)])
            
#             #Check if the current element is adjacent to any no_flip 'O'
#             if all_os[i] in adjacents:
#                 no_flip.append(all_os.pop(i))
#                 flipped = True
#                 i = 0
#                 continue

#             i += 1
#             flipped = False


#         # Rewritting the board
#         #   Resetting the board to all "X"
#         for i in range(M):
#             board[i] = ["X"]*N
        
#         #   preserving the no_flip 'O's
#         for o in no_flip:
#             board[o[0]][o[1]] = 'O'

#     # Testing
#     solve(board=board)

#     'This solution met 98.2% of the cases'


#     'DFS Approach'
#     def solve(board):

#         n,m=len(board),len(board[0])
#         seen=set()

#         def is_valid(i,j):
#             return 0 <= i < n and 0<= j <m and board[i][j]=="O" and (i,j) not in seen
        
#         def is_border(i,j):
#             return i == 0 or i == n-1 or j == 0 or j == m-1
        
#         def dfs(i,j):

#             board[i][j]="y"
#             seen.add((i,j))

#             for dx , dy in ((0,1) ,(0,-1) ,(1,0),(-1,0)):
#                 new_i , new_j = dx + i , dy + j

#                 if is_valid(new_i , new_j):
#                     dfs(new_i , new_j)
            
#         for i in range(n):
#             for j in range(m):
#                 if is_border(i,j) and board[i][j]=="O":
#                     dfs(i,j) 
                    
#         for i in range(n):
#             for j in range(m):
#                 if board[i][j]=="y":
#                     board[i][j]="O"
#                 else:
#                     board[i][j]="X"

#     # Testing
#     solve(board)

#     'Done'

'''200. Number of Islands'''
# def x():

#     # Input
#     # Case 1
#     grid = [
#       ["1","1","1","1","0"],
#       ["1","1","0","1","0"],
#       ["1","1","0","0","0"],
#       ["0","0","0","0","0"]
#     ]
#     # Ouput: 1

#     # Case 2
#     grid = [
#       ["1","1","0","0","0"],
#       ["1","1","0","0","0"],
#       ["0","0","1","0","0"],
#       ["0","0","0","1","1"]
#     ]
#     # Ouput: 3

#     # Custom Case
#     grid = [
#         ["1","0"]
#         ]
#     # Ouput: 1


#     'My BFS Approach'
#     def numIslands(grid:list[list[str]]) -> int:
        
#         if len(grid) == 1:
#             return len([x for x in grid[0] if x =='1'])

#         # Create the 'lands' coordinates
#         coord = []

#         # Collecting the 'lands' coordinates
#         for i, row in enumerate(grid):
#             coord.extend((i, j) for j, value in enumerate(row) if value == '1')


#         # Create the groups holder
#         islands = []
#         used = set()


#         # BFS Definition
#         def bfs(root:tuple) -> list:

#             queue = [root]
#             curr_island = []

#             while queue:

#                 land = queue.pop(0)
#                 x, y = land[0], land[1]
                
#                 if grid[x][y] == '1' and (land not in curr_island and land not in used):

#                     curr_island.append(land)
                
#                     # Define next lands to search
#                     if x == 0:
#                         if y == 0:
#                             next_lands = [(x+1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x+1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x+1,y),(x,y-1)]
                    
#                     elif x < len(grid)-1:
#                         if y == 0:
#                             next_lands = [(x-1,y),(x+1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x-1,y),(x+1,y),(x,y-1)]
                    
#                     else:
#                         if y == 0:
#                             next_lands = [(x-1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x-1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x-1,y),(x,y-1)]
                                    
#                     # List the next lands to visit
#                     for next_land in next_lands:

#                         if next_land not in curr_island:

#                             queue.append(next_land)

#             return curr_island
            

#         # Checking all the 1s in the grid
#         for elem in coord:

#             if elem not in used:

#                 island = bfs(elem)

#                 islands.append(island)
#                 used.update(set(island))
        
#         return len(islands)

#     # Testing
#     print(numIslands(grid=grid))
    
#     'Note: This could be done way simplier'


#     'Simplified & Corrected BFS Approach'
#     def numIslands(grid:list[list[str]]) -> int:

#         if not grid:
#             return 0

#         num_islands = 0
#         directions = [(1,0),(-1,0),(0,1),(0,-1)]

#         for i in range(len(grid)):

#             for j in range(len(grid[0])):

#                 if grid[i][j] == '1':

#                     num_islands += 1

#                     queue = [(i,j)]

#                     while queue:

#                         x, y = queue.pop(0)

#                         if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':

#                             grid[x][y] = '0'    # Mark as visited

#                             for dx, dy in directions:

#                                 queue.append((x + dx, y + dy))
        
#         return num_islands

#     'Done'

'''212. Word Search II'''
# def x():

#     # Input
#     # Case 1
#     board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
#     words = ["oath","pea","eat","rain"]
#     # Output: ["eat","oath"]

#     # Case 2
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []

#     # Custom Case
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []


#     '''
#     My Approach
    
#         Intuiton:

#             - Based on the 'Word Seach I' backtracking solution, I will try to emulate the same but
#                 since now there are multiple word to lookout for, I will rely on a Trie implementation
#                 to look out for prefixes to optimize the process.

#                 And to try to make it work, I will pull the first letter of each word and only start
#                 the searches from those positions, so, roughly the plan is:

#                 1. Collect the coordinates of the first letter from each of the word and store them in a dict
#                     as {'word': coordinates[(x,y)]}, if a word has no coordinates and it means it won't be found
#                     in the matrix, so it won't be in Trie.
                
#                 2. Initiate the Trie with the words with coordinates.

#                 3. Iterate through each of the words, and iterate for each pair of coordinates to look out for that word,
#                     if found, add it to a result list if don't pass to the next pair of coordinates, and so on for each word.
                
#                 4. Return the found words.
#     '''

#     'ACTUAL CODE'
#     # TRIE IMPLEMENTATION

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False


#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()
                
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#         def search(self, word:str) -> bool:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return curr_node.is_word

#         def stars_with(self, prefix:str) -> bool:

#             curr_node = self.root

#             for char in prefix:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         import copy

#         #AUX BACKTRACK FUNC DEF
#         def backtrack(i:int, j:int, k:str) -> bool:

#             if new_trie.search(k):
#                 return True
                    
#             if not new_trie.stars_with(k):
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = '.'

#             #1
#             if 0<i<len(board)-1 and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #2
#             elif 0 == i and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #3
#             elif 0 == i and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #4
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #5
#             elif 0<i<len(board)-1 and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #6
#             elif 0<i<len(board)-1 and len(board[0])-1 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #7
#             elif len(board)-1 == i and 0 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
            
#             #8
#             elif len(board)-1 == i and 0<j<len(board[0])-1:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True

#             #9
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True


#             board[i][j] = temp

#             return False 
        

#         # COLLECT FIRST LETTER COORDINATES FOR EACH WORD
#         words_dict = {}

#         for word in words:

#             coordinates = []

#             for i,row in enumerate(board):
#                 coordinates.extend([(i,j) for j,elem in enumerate(row) if board[i][j] == word[0]])

#             if coordinates:
#                 words_dict[word] = coordinates


#         # INITIATE THE TRIE
#         new_trie = Trie()

#         for word in words_dict.keys():
#             new_trie.insert(word)

#         x = 0

#         result = []

#         # ITERATE THE DICT
#         for word in words_dict:

#             temp_board = copy.deepcopy(board)

#             for i,j in words_dict[word]:

#                 if backtrack(i, j, word[0]):

#                     result.append(word)
#                     board = temp_board

#         return result

#     # Testing
#     print(findWords(board=board, words=words))

#     '''
#     Notes:
#         My solution and approach wasn't that far. The logic was correct, the execution was the one to fail.
#         My version of the solution tends to get redundant and can't handle efficiently larger inputs
#     '''

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False

#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:
#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()            
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         # Build the Trie
#         trie = Trie()

#         for word in words:
#             trie.insert(word)
        
#         # Auxiliary vars
#         rows, cols = len(board), len(board[0])
#         result = set()
#         visited = set()

#         #Aux DFS Func
#         def dfs(node:TrieNode, i:int, j:str, path:str) -> None:

#             if i<0 or i>=rows or j<0 or j>=cols or (i,j) in visited or board[i][j] not in node.values:
#                 return
            
#             visited.add((i,j))
#             node = node.values[board[i][j]]
#             path += board[i][j]

#             if node.is_word:
#                 result.add(path)
#                 node.is_word = False    # To avoid duplicate results

#             # Explore neighbors in 4 directions (up, down, left, right)
#             for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
#                 dfs(node, x, y, path)
            
#             visited.remove((i,j))        

#         # Traverse the board
#         for i in range(rows):
#             for j in range(cols):
#                 dfs(trie.root, i, j, '')        

#         return result

#     'Done'

'''240. Search a 2D Matrix II'''
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#     target = 5
#     # Output: True

#     # Case 2
#     matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#     target = 20
#     # Output: False


#     '''
#     My Approach

#         Intuition:

#             - Iterativelly search in the first row of the matrix if the value is in there by a belonging test
#                 - If the value is in the element, break and return True / else, pop that element from the matrix
#                 - Transpose the matrix and start over until there's no more elements in the matrix
#             - If the loop reaches the last element of the matrix, return False
#     '''

#     def search_matrix(matrix:list[list[int]], target: int) -> bool:

#         m = len(matrix)
#         n = len(matrix[0])

#         # Handle the corner case
#         if n == m == 1:
#             return target == matrix[0][0]

#         while matrix:

#             # print(matrix)

#             element = matrix.pop(0)

#             if target in element:
#                 return True
            
#             matrix = [list(x) for x in zip(*matrix)]
        
#         return False

#     # Testing
#     print(search_matrix(matrix=matrix,target=target))

#     '''Note: This approach doesn't worked because dinamically changing the data structure mess in how python checks membership'''


#     'Binary search approach'
#     def search_matrix(matrix:list[list[int]], target: int) -> bool:

#         m = len(matrix)
#         n = len(matrix[0])

#         # Handle the corner case
#         if n == m == 1:
#             return target == matrix[0][0]

#         row, col = m-1, 0   # Start the search from the bottom left corner

#         while row >= 0 and col < n:

#             element = matrix[row][col]

#             if element == target:
#                 return True
            
#             elif element > target:
#                 row -= 1
                    
#             else:
#                 col += 1
        
#         return False

#     # Testing
#     print(search_matrix(matrix=matrix,target=target))

#     'done'

'''289. Game of Life'''
# def x():

#     # Input
#     # Case 1
#     board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
#     # Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]

#     # Case 2
#     board = [[1,1],[1,0]]
#     # Output: [[1,1],[1,1]]
    

#     '''
#     My Approach

#         Intuition:
#             - Create a 'result matrix' filled with 0s with the size of the original one.
#             - Create a auxiliary function that evaluates all the neighbors to collect the number of 1s to apply the rule.
#                 + In the neighbors evaluation since there are 1s and 0s, to avoid more looping I will sum up to get the number of living cells.
#             - Populate the result matrix according to the rules.
#     '''

#     def game_of_life(board:list[list[int]]) -> None:

#         # Set the matrix dimentions
#         m,n = len(board),len(board[0])

#         # Define the holder matrix
#         holder = [[0]*n for _ in range(m)]

#         # Define the directions of the neighbors
#         directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

#         # Iterate to evaluate each of the cells of the original board
#         for i in range(m):

#             for j in range(n):

#                 # Define the actual neighbors
#                 neighbors = [ board[i+dx][j+dy] for dx, dy in directions if 0 <= i+dx < m and 0 <= j+dy < n ]

#                 # Evalue the number of live neighbors
#                 neighbors = sum(neighbors)

#                 # APPLY THE RULES
#                 if board[i][j] == 1:
                    
#                     # 1. Any live cell with fewer than two live neighbors dies as if caused by under-population.
#                     if neighbors < 2:
#                         holder[i][j] = 0    # Update the holder matrix in the exact position with the result of the rule apply
                    
#                     # 2. Any live cell with two or three live neighbors lives on to the next generation.
#                     elif 2 <= neighbors <= 3:
#                         holder[i][j] = 1
                    
#                     # 3. Any live cell with more than three live neighbors dies, as if by over-population.
#                     elif neighbors > 3:
#                         holder[i][j] = 0

#                 else:
#                     # 4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
#                     if neighbors == 3:
#                         holder[i][j] = 1


#         # Prepare the output: Modify the original board according to the result of the game
#         for i in range(m):
#             for j in range(n):
#                 board[i][j] = holder[i][j] 


#     # Testing
#     for i in range(len(board)):
#         print(board[i])

#     game_of_life(board=board)
#     print()

#     for i in range(len(board)):
#         print(board[i])

#     'Done: My approach worked and beated 72% of submissions in runtime and 35% in space.'




'''51. N-Queens'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     n = 4
#     # Output: [
#     #   [".Q..","...Q","Q...","..Q."],
#     #   ["..Q.","Q...","...Q",".Q.."]
#     # ]

#     # Case 2
#     n = 1
#     # Output: [Q]
    
#     # Case 2
#     n = 9
#     # Output: [Q]

#     '''
#     Explanation
        
#         The "N-Queens" problem is a classic example of a backtracking algorithm challenge. In this problem, you are asked to place N queens on an N x N chessboard 
#         such that no two queens attack each other. Queens can attack other queens that are in the same row, column, or diagonal.

        
#         Problem breakdown

#             1. You need to place N queens on an N x N board.
#             2. No two queens can be placed on the same row, column, or diagonal.
#             3. The goal is to return all possible arrangements that satisfy the above conditions.

        
#         Approach: Backtracking
            
#             We will solve this using backtracking, which is an algorithmic technique for solving constraint satisfaction problems. 
#             The idea is to place queens one by one and check for validity at each step. If a solution is found, we record it; otherwise, 
#             we backtrack and try different placements.
        
        
#         Steps:

#             1. Place Queens Row by Row: Start placing queens from the first row to the last. At each row, try placing a queen in each column, one by one, and check if it’s a valid position.

#             2. Check for Conflicts: After placing a queen, check whether it’s under attack from previously placed queens (i.e., check if there is a conflict in columns or diagonals).

#             3. Backtrack: If a conflict is found, remove the queen and try placing it in a different column. If no valid position exists in a row, backtrack to the previous row and move that queen.

#             4. Store Valid Solutions: When all queens are placed successfully, store the board configuration.
#     '''

#     def solveNQueens(n):

#         def is_valid(board, row, col):

#             # Check the column
#             for i in range(row):
#                 if board[i] == col:
#                     return False
                
#             # Check the diagonal (left-up)
#             for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
#                 if board[i] == j:
#                     return False
                
#             # Check the diagonal (right-up)
#             for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
#                 if board[i] == j:
#                     return False
                
#             return True


#         def backtrack(row, board, solutions):

#             if row == n:

#                 # If all queens are placed, convert the board to the required output format
#                 solutions.append(["." * col + "Q" + "." * (n - col - 1) for col in board])
#                 return
            
#             for col in range(n):

#                 if is_valid(board, row, col):
                    
#                     board[row] = col
#                     backtrack(row + 1, board, solutions)
#                     board[row] = -1  # Backtrack


#         # Initialize board and solutions list
#         solutions = []
#         board = [-1] * n  # Keeps track of queen's position in each row
#         backtrack(0, board, solutions)

#         return solutions


#     # Testing
#     solution = solveNQueens(n=n)
#     print(f'# of Solution: {len(solution)}')
#     for i in solution[1]:
#         print(i)

#     '''Note: Done'''

'''64. Minimum Path Sum'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     grid = [[1,3,1],[1,5,1],[4,2,1]]
#     # Output: 7

#     # Case 2
#     grid = [[1,2,3],[4,5,6]]
#     # Output: 12

#     '''
#     My Approach (Dynamic Programming)

#         Intuition:
            
#             - Initialize a 'result' grid of dimentions m+1, n+1 and initilize the first row and the first column of each rown in 201*.
#             - Start traversing 'result' from result[1][1] and the operation will be to assign that cell: max( grid[i-1][j-1]+result[i][j-1], grid[i][j]+result[i-1][j1] )
#                 Except for the result[1][1], which will be assigned result[0][0]
#             - Return result[-1][-1] which will contain the shortest path to get there.

#             *201 is assigned as the value above the contrain '0 <= grid[i][j] <= 200', to say 'don't select this one'
#     '''

#     def minPathSum(grid: list[list[int]]) -> int:

#         # Capture the input dimentions
#         m, n = len(grid), len(grid[0])

#         # Handle Corner case: 1x1 input
#         if m == n == 1:
#             return grid[0][0]
        
#         # Handle Corner case: 1xn input
#         if m == 1:
#             return sum(grid[0])
        
#         # Handle Corner case: mx1 input
#         if m == 1:
#             return sum([x[0] for x in grid])
        

#         # Initialize the result grid holder
#         result = [[0]*(n+1) for _ in range(m+1)]

#         # Set the first row to 201
#         for col in range(len(result[0])):
#             result[0][col] = 201
        
#         # Set the first column to 201
#         for row in range(len(result)):
#             result[row][0] = 201

#         # Traverse the 'result' grid starting from [1][1]
#         for i in range(1, len(result)):
#             for j in range(1, len(result[0])):

#                 if j == i == 1:
#                     result[i][j] = grid[i-1][j-1]

#                 else:
#                     result[i][j] = min( grid[i-1][j-1]+result[i][j-1], grid[i-1][j-1]+result[i-1][j] )

#         # Return the last cell which is the one containing the shortest path
#         return result[-1][-1]

#     # Testing
#     print(minPathSum(grid=grid))

#     '''Note: My approach solved 96% of testcases'''


#     '''
#     My Approach (Dynamic Programming) - Correction
           
#         The initial idea was not wrong, but initializing with extra row and column could lead to miscalculations. 
#         So the 'result' holder will initialize only with the size of the grid and operation will be:
#             - For the first row: only summing up itself with the cell at the left.
#             - For the first column: only summing up itself with the cell at the top.
#             - For the rest of the result grid: Sum as planned.
#     '''


#     def minPathSum(grid: list[list[int]]) -> int:

#         # Capture the input dimentions
#         m, n = len(grid), len(grid[0])

#         # Handle Corner case: 1x1 input
#         if m == n == 1:
#             return grid[0][0]


#         # Initialize the result grid holder
#         result = [[0]*(n) for _ in range(m)]

#         # Initilize the top left cell of the result grid
#         result[0][0] = grid[0][0]


#         # Traverse the first row
#         for col in range(1, len(result[0])):
#             result[0][col] = result[0][col-1] + grid[0][col]
        
#         # Traverse the first column
#         for row in range(1, len(result)):
#             result[row][0] = result[row-1][0] + grid[row][0]

#         # Traverse the rest of 'result' grid starting from [1][1]
#         for i in range(1, len(result)):
#             for j in range(1, len(result[0])):
#                 result[i][j] = min( grid[i][j]+result[i][j-1], grid[i][j]+result[i-1][j] )

#         # Return the last cell which is the one containing the shortest path
#         return result[-1][-1]

#     # Testing
#     print(minPathSum(grid=grid))

#     '''Notes: Done!'''

'''74. Search a 2D Matrix'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
#     target = 3
#     # Output: True

#     # Case 2
#     matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
#     target = 13
#     # Output: False

#     '''
#     My Approach

#        Apply the Binary Search idea on a matrix.
                        
#     '''

#     def searchMatrix(matrix: list[list[int]], target: int) -> bool:

#         # Handle Corner case: Target out of boundaries
#         if target < matrix[0][0] or target > matrix[-1][-1]:
#             return False
        
#         # Define the two pointers to binary search the target
#         low, high = 0, len(matrix)

#         # Start the binary search
#         while low < high:

#             # Define the mid pointer
#             mid = (low + high) // 2

#             # If the target is present in the middle element
#             if target in matrix[mid]:
#                 return True
            
#             # If the item is greater than the last item of the middle element
#             elif target > matrix[mid][-1]:

#                 # Redefine the low pointer
#                 low = mid + 1
            
#             # If the item is smaller than the first item of the middle element
#             elif target < matrix[mid][0]:

#                 # Redefine the high pointer
#                 high = mid
            
#             # If no condition is met, return False
#             else:
#                 return False

#         # If the item was not found in the loop, it means it not present in the matrix
#         return False

#     # Testing
#     print(searchMatrix(matrix=matrix, target=target))

#     '''Note: It worked right away! the results were: 6.27% in Runtime & 92.27% in Memory'''













