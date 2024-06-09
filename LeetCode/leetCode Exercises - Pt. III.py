'''
CHALLENGES INDEX

118. Pascal's Triangle
121. Best Time to Buy and Sell Stock
122. Best Time to Buy and Sell Stock II
124. Binary Tree Maximum Path Sum
125. Valid Palindrome
127. Word Ladder
128. Longest Consecutive Sequence
130. Surrounded Regions
131. Palindrome Partitioning
134. Gas Station
138. Copy List with Random Pointer
139. Word Break
140. Word Break II
141. Linked List Cycle
146. LRU Cache
148. Sort List
149. Max Points on a Line
150. Evaluate Reverse Polish Notation
152. Maximum Product Subarray
155. Min Stack
160. Intersection of Two Linked Lists

(21)

'''




'''118. Pascal's Triangle'''

#Input

# #case 1
# numRows = 5
# #Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]

# #case 2
# numRows = 1
# #Output: [[1]]

# My Approach

'''
Intuition:
    initialize a preset solution to [[1],[1,1]] and according to the
    parameter passed in the function, start to sum and populate this sums to a list
    like [1]+[resulting_sums]+[1] and return that back to the preset solution, to operate over that
    new element,

        The number of loops will be numRows - 2 (given the 2 initial elements)
'''

# def generate(numRows:int) -> list[list[int]]:

#     result = [[1],[1,1]]

#     if numRows == 1:
#         return [result[0]]
    
#     if numRows == 2:
#         return result
    

#     for i in range(1, numRows-1):

#         new_element = []

#         for j in range(i):
#             new_element.append(result[-1][j]+result[-1][j+1])

#         if new_element:
#             result.append([1]+new_element+[1])

#     return result

# print(generate(numRows=5))

'It worked!'




'''121. Best Time to Buy and Sell Stock'''

# Input

# #Case 1
# prices = [7,1,5,3,6,4]
# #Output: 5

# #Case 2
# prices = [7,6,4,3,1]
# #Output: 0



# My approach

'''
Intuition
    - Corner Case: if is a ascendingly sorted list, return 0.
    
    - Pick the first item and set the profit as the max between the current profit and the difference between the first element
      the max value from that item forward.
    
    Do this in a while loop until len(prices) = 1.
'''


# def maxProfit(prices: list[int]) -> int:

#     profit = 0

#     if prices == sorted(prices, reverse=True):
#         return profit
    

#     while len(prices) > 1:

#         purchase = prices.pop(0)

#         profit = max(profit, max(prices)-purchase)
    
#     return profit


# print(maxProfit(prices=prices))

'This approach met 94% of the results'


# Kadane's Algorithm

# def maxProfit(prices: list[int]) -> int:

#     buy = prices[0]
#     profit = 0

#     for num in prices[1:]:

#         if num < buy:
#             buy = num
        
#         elif num-buy > profit:
#             profit = num - buy
    
    
#     return profit  



# print(maxProfit(prices=prices))

'Done'




'''122. Best Time to Buy and Sell Stock II'''

#Input

# #Case 1
# prices = [7,1,5,3,6,4]
# #Output: 7

# #Case 2
# prices = [1,2,3,4,5]
# #Output: 4

# #Case 3
# prices = [7,6,4,3,1]
# #Output: 0

# #Custom Case
# prices = [3,3,5,0,0,3,1,4]
# #Output: 0

# # My approach
# def maxProfit(prices:list[int]) -> int:

#     if prices == sorted(prices, reverse=True):
#         return 0
    
#     buy = prices[0]
#     buy2 = None
#     profit1 = 0
#     profit2 = 0
#     total_profit = 0

#     for i in range(1, len(prices)):

#         if prices[i] < buy:
#             buy = prices[i]
        
#         elif prices[i] - buy >= profit1:            
#             profit1 = prices[i] - buy
#             buy2 = prices[i] 

#             for j in range(i+1, len(prices)):

#                 if prices[j] < buy2:
#                     buy2 = prices[j]

#                 elif prices[j] - buy2 >= profit2:
#                     profit2 = prices[j] - buy2
#                     total_profit = max(total_profit, profit1 + profit2)
        
#         total_profit = max(total_profit, profit1)

#     return total_profit


# print(maxProfit(prices=prices))

'This solution went up to solve 83% of the cases, the gap was due to my lack of understanding of the problem'


# # Same Kadane's but modified
# def maxProfit(prices:list[int]) -> int:

#     max = 0 
#     start = prices[0]
#     len1 = len(prices)

#     for i in range(0 , len1):

#         if start < prices[i]: 
#             max += prices[i] - start

#         start = prices[i]

#     return max


# print(maxProfit(prices=prices))

'My mistake was to assume it can only be 2 purchases in the term, when it could be as many as it made sense'




'''124. Binary Tree Maximum Path Sum'''

# # Base 
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

#Input

# #Case 1
# tree_layout = [1,2,3]
# root = TreeNode(val=1, left=TreeNode(val=2), right=TreeNode(val=3))
# #Output: 6

# #Case 2
# tree_layout = [-10,9,20,None, None,15,7]
# left = TreeNode(val=9)
# right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))
# root = TreeNode(val=-10, left=left, right=right)
# #Output: 42

# #Custom Case
# tree_layout = [1,-2,3,1,-1,-2,-3]
# left = TreeNode(val=-2, left=TreeNode(val=1), right=TreeNode(val=3))
# right = TreeNode(val=-3, left=TreeNode(val=-2, left=TreeNode(val=-1)))
# root = TreeNode(val=1, left=left, right=right)
# #Output: 3


#My approach

'''
Intuition:
    - Make a preorder traversal tree list.
    - Apply Kadane's algorithm to that list.
'''

# def maxPathSum(root:TreeNode) -> int:

#     #First, Preorder
#     path = []

#     def preorder(node:TreeNode) -> None:

#         if node:
#             preorder(node=node.left)
#             path.append(node.val)
#             preorder(node=node.right)

#     preorder(node=root)

#     #Now Kadane's
#     max_so_far = max_end_here = path[0]

#     for num in path[1:]:

#         max_end_here = max(num, max_end_here + num)
#         max_so_far = max(max_so_far, max_end_here)

#     return max_so_far


# print(maxPathSum(root=root))

'''
Notes:
    - On the first run it went up to 59% of the cases, thats Kudos for me! :D
    - The problem with this algorithm is that it supposes that after reaching a parent and child node,
      it's possible to go from a right child to the parent of the parent and that either forcibly makes
      to pass twice from the parent before going to the granparent, or that one grandchild is connected
      to the grandfather, which is also out of the rules.

      I misinterpret this because one of the examples showed a path [leftchild, parent, rightchild] which
      is valid only if we don't want to pass thruough the grandparent.
    
    The best choice here is to make a recursive proning algorithm
'''


# #A recursive approach
# def maxPathSum(root):

#     max_path = float('-inf') #Placeholder

#     def get_max_gain(node):

#         nonlocal max_path

#         if not node:
#             return 0
        
#         gain_on_left = max(get_max_gain(node.left),0)
#         gain_on_right = max(get_max_gain(node.right),0)

#         current_max_path = node.val + gain_on_left + gain_on_right
#         max_path = max(max_path, current_max_path)

#         return node.val + max(gain_on_left, gain_on_right)
    
#     get_max_gain(root)

#     return max_path

# print(maxPathSum(root))
'Done'




'''125. Valid Palindrome'''

# def isPalindrome(s:str) -> bool:

#     s = ''.join([x for x in s if x.isalpha()]).casefold()

#     return s == s[::-1]



# a = '0P'

# a = ''.join([x for x in a if x.isalnum()]).casefold()

# print(a)
'Done'




'''127. Word Ladder'''

#Input

# #Case 1
# begin_word, end_word, word_list = 'hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']
# #Output: 5

# #Custom Case
# begin_word, end_word, word_list = 'a', 'c', ['a', 'b', 'c']
# #Output: 5


# My approach

'''
Intuition:
    1. handle the corner case: the end_word not in the word_list
    2. create an auxiliary func that check the word against the end_word: True if differ at most by 1 char, else False.
    3. create a counter initialized in 0
    4. start checking the begin_word and the end_word, if False sum 1 to the count, and change to the subquent word in the word_list and do the same.
'''

# def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#     if endWord not in wordList:
#         return 0
    
#     def check(word):
#         return False if len([x for x in word if x not in endWord]) > 1 else True
       
#     if beginWord not in wordList:
#         wordList.insert(0,beginWord)
#         count = 0
    
#     else:
#         count = 1
    
#     for elem in wordList:
#         count += 1

#         if check(elem):
#             return count     
            
#     return 0


# print(ladderLength(beginWord=begin_word, endWord=end_word, wordList=word_list))


'This solution only went up to the 21% of the cases'


# bfs approach

# from collections import defaultdict, deque

# def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#     if endWord not in wordList or not endWord or not beginWord or not wordList:
#         return 0

#     L = len(beginWord)
#     all_combo_dict = defaultdict(list)

#     for word in wordList:
#         for i in range(L):
#             all_combo_dict[word[:i] + "*" + word[i+1:]].append(word) 

#     queue = deque([(beginWord, 1)])
#     visited = set()
#     visited.add(beginWord)

#     while queue:
#         current_word, level = queue.popleft()

#         for i in range(L):
#             intermediate_word = current_word[:i] + "*" + current_word[i+1:]

#             for word in all_combo_dict[intermediate_word]:

#                 if word == endWord:
#                     return level + 1

#                 if word not in visited:
#                     visited.add(word)
#                     queue.append((word, level + 1))
                    
#     return 0
'Done'




'''128. Longest Consecutive Sequence'''

#Input

# #Case 1
# nums = [100,4,200,1,3,2]
# #Output: 4

# #Case 2
# nums = [0,3,7,2,5,8,4,6,0,1]
# #Output: 9


#My approach

# def longestConsecutive(nums:list)->int:

#     if not nums:
#         return 0
   
#     nums.sort()

#     sequences = {}

#     for i in range(len(nums)):

#         curr_seqs = [x for elem in sequences.values() for x in elem]

#         if nums[i] not in curr_seqs:

#             sequences[nums[i]] = [nums[i]]

#             for j in range(i+1,len(nums)):
                
#                 criteria = range( min(sequences[nums[i]])-1, max(sequences[nums[i]])+2)
#                 if nums[j] in criteria:
#                     sequences[nums[i]].append(nums[j])

#     result = max(sequences.values(), key=len)

#     return len(set(result))

# print(longestConsecutive(nums=nums))

'This solution went up to 83% of the cases'


# Another Approach

# def longestConsecutive (nums):

#     if not nums:
#         return 0
    
#     num_set = set(nums)

#     longest = 1

#     for num in nums:

#         count = 1

#         if num-1 not in num_set:

#             x = num

#             while x+1 in num_set:
               
#                 count+=1
#                 x+=1

#         longest = max(longest, count)

#     return longest

# print(longestConsecutive(nums=nums))

'Done'




'''130. Surrounded Regions'''

#Input

# #Case 1
# board = [
#     ["X","X","X","X"],
#     ["X","O","O","X"],
#     ["X","X","O","X"],
#     ["X","O","X","X"]
#     ]
# # output = [
# #     ["X","X","X","X"],
# #     ["X","X","X","X"],
# #     ["X","X","X","X"],
# #     ["X","O","X","X"]
# #     ]

# #Case 2
# board = [
#     ['X']
#     ]
# # output = [
#     # ['X']
#     # ]

# #Custom Case
# board = [["O","O"],["O","O"]]



#My approach

'''
Intuition:
    1. Check if there is any 'O' at the boarders.
    2. Check is there is any 'O' adjacent to the one in the boarder:
        - If do, add them to the not-be-flipped ground and re run.
        - if doesn't, flip everything to 'X' and return
    (Do this until there is no 'O' unchecked )
'''

# def solve(board:list[list[str]]) -> None:

#     M = len(board)
#     N = len(board[0])

#     no_flip = []
#     all_os = []


#     # Collect all 'O's
#     for i in range(M):
#         all_os.extend((i,j) for j in range(N) if board[i][j] == 'O')
    

#     #   Check if there is a boarder 'O' within the group
#     for i in range(len(all_os)):

#         if all_os[i][0] in (0, M-1) or all_os[i][1] in (0, N-1):
#             no_flip.append(all_os[i])


#     # Collect the 'O's near to no_flip 'O' iteratively
#     flipped = None
#     i = 0

#     while True:

#         # Condition to end the loop
#         if len(all_os) == 0 or i == len(all_os) and flipped is False:
#             break

#         #Collecting the possibilities of an adjacent 'O'
#         adjacents = []

#         for pos in no_flip:
#             adjacents.extend([(pos[0]-1, pos[1]), (pos[0]+1, pos[1]), (pos[0], pos[1]-1), (pos[0], pos[1]+1)])
        
#         #Check if the current element is adjacent to any no_flip 'O'
#         if all_os[i] in adjacents:
#             no_flip.append(all_os.pop(i))
#             flipped = True
#             i = 0
#             continue

#         i += 1
#         flipped = False


#     # Rewritting the board
#     #   Resetting the board to all "X"
#     for i in range(M):
#         board[i] = ["X"]*N
    
#     #   preserving the no_flip 'O's
#     for o in no_flip:
#         board[o[0]][o[1]] = 'O'


# solve(board=board)

'This solution met 98.2% of the cases'


#DFS Approach

# def solve(board):

#     n,m=len(board),len(board[0])
#     seen=set()

#     def is_valid(i,j):
#         return 0 <= i < n and 0<= j <m and board[i][j]=="O" and (i,j) not in seen
    
#     def is_border(i,j):
#         return i == 0 or i == n-1 or j == 0 or j == m-1
    
#     def dfs(i,j):

#         board[i][j]="y"
#         seen.add((i,j))

#         for dx , dy in ((0,1) ,(0,-1) ,(1,0),(-1,0)):
#             new_i , new_j = dx + i , dy + j

#             if is_valid(new_i , new_j):
#                 dfs(new_i , new_j)
        
#     for i in range(n):
#         for j in range(m):
#             if is_border(i,j) and board[i][j]=="O":
#                 dfs(i,j) 
                
#     for i in range(n):
#         for j in range(m):
#             if board[i][j]=="y":
#                 board[i][j]="O"
#             else:
#                 board[i][j]="X"

# solve(board)

'Done'




'''131. Palindrome Partitioning'''

#Input

# # Case 1
# s = 'aab'
# # Output: [["a","a","b"],["aa","b"]]

# # Custom Case
# s = 'aabcdededcbada'
# # Output: [["a","a","b"],["aa","b"]]



# My approach

'''
Intuition:

    Here I don't actually have much ideas in how to solve it, but one good approach
    I think woul dbe to make a function that can pull all the palindroms present in a string.

    that could be a good start point.
'''

# # Custom Case
# s = 'aabcdededcbada'
# # Output: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 

# def palindromes(string:str) -> list[str]:

#     s_len = len(string)
#     palindromes = []

#     for i in range(s_len, 1, -1):   # from s_len down to length 2 of substring
       
#         j = 0

#         while j + i <= s_len: 

#             subs = string[j:j+i]

#             if subs == subs[::-1]:

#                 palindromes.append(subs)

#             j += 1

#     print(palindromes)



# # Printout: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 
# palindromes(string=s)

'''
At least this I was able to do, from here on, I am feeling I am going to brute forcing this and it won't end up being efficient.

I didn't actually solved it but I don't want to waste more time over this
'''




'''134. Gas Station'''

# Input

# #Case 1
# gas, cost = [1,2,3,4,5], [3,4,5,1,2]
# #Output = 3

# #Case 2
# gas, cost = [2,3,4], [3,4,3]
# #Output = -1

# # #Custom Case 
# gas, cost = [3,1,1], [1,2,2]
# #Output = 0


# My Approach

'''
Intuition:
    - Handle the corner case where sum(gas) < sum(cos) / return -1
    - Collect the possible starting point (Points where gas[i] >= cost[i])
    - Iterate to each starting point (holding it in a placeholder) to check 
        if a route starting on that point completes the lap:
        
        - if it does: return that starting point
        - if it doesn't: jump to the next starting point

    - If no lap is completed after the loop, return -1.

'''

# def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:

    
#     # Handle the corner case
#     if sum(gas) < sum(cost):
#         return -1
    

#     # Collect the potential starting stations
#     stations = [i for i in range(len(gas)) if gas[i] >= cost[i]]


#     # Checking routes starting from each collected station
#     for i in stations:

#         station = i
#         tank = gas[i]

#         while tank >= 0:
            
#             # Travel to the next station
#             tank = tank - cost[station] 

#             # Check if we actually can get to the next station with current gas
#             if tank < 0:
#                 break
                
#             # If we are at the end of the stations (clockwise)
#             if station + 1 == len(gas):
#                 station = 0
                        
#             else:
#                 station += 1
                        
#             #If we success in making the lap
#             if station == i:
#                 return i
        
#             # Refill the tank
#             tank = tank + gas[station]


#     # in case no successful loop happens, return -1
#     return -1

# print(canCompleteCircuit(gas=gas, cost=cost))

'My solution met 85% of the test cases'


# # Another approach

# def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:

    
#     # Handle the corner case
#     if sum(gas) < sum(cost):
#         return -1
    
#     current_gas = 0
#     starting_index = 0

#     for i in range(len(gas)):

#         current_gas += gas[i] - cost[i]

#         if current_gas < 0:
#             current_gas = 0
#             starting_index = i + 1
            
#     return starting_index

# print(canCompleteCircuit(gas=gas, cost=cost))

'This simplified version prooved to be more efficient'




'''138. Copy List with Random Pointer'''

# # Base
# class Node:
#     def __init__(self, x, next=None, random=None):
#         self.val = int(x)
#         self.next = next
#         self.random = random


# #Input

# #Case 1
# head_map = [[7,None],[13,0],[11,4],[10,2],[1,0]]

# #Build the relations of the list
# nodes = [Node(x=val[0]) for val in head_map]

# for i in range(len(nodes)):
#     nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#     nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

# head = nodes[0]

#Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


# #Case 2
# head_map = [[1,1],[2,1]]

# #Build the relations of the list
# nodes = [Node(x=val[0]) for val in head_map]

# for i in range(len(nodes)):
#     nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#     nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

# head = nodes[0]

# #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


# #Case 3
# head_map = [[3,None],[3,0],[3,None]]

# #Build the relations of the list
# nodes = [Node(x=val[0]) for val in head_map]

# for i in range(len(nodes)):
#     nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#     nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

# head = nodes[0]

# #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]



# My Approach

'''
Intuition:
    - Traverse through the list
    - Create a copy of each node and store it into a list along side with the content of the random pointer.
    - Traverse the list linking each node to the next and the random pointer to the position in that list.

Thoughts:

- It is possible to create the list with a recursive solution but it'll be still necesary to traverse again
    to collect the content of the random pointer or how else I can point to somewhere at each moment I don't know if it exist. 

'''

# def copyRandomList(head:Node) -> Node:

#     # Handle the corner case where there is a single node list
#     if head.next == None:
#         result = Node(x = head.val, random=result)
#         return result

#     # Initilize a nodes holder dict to collect the new nodes while traversing the list
#     nodes = {}

#     # Initilize a nodes holder list to collect the old nodes values while traversing the list
#     old_nodes_vals = []

#     # Initialize a dummy node to traverse the list
#     current_node = head

#     # Traverse the list
#     while current_node is not None:

#         # Collect the old nodes
#         old_nodes_vals.append(current_node.val)

#         # Check if the node doesn't already exist due to the random pointer handling
#         if current_node.val not in nodes.keys(): 

#             new_node = Node(x = current_node.val)
#             nodes[new_node.val] = new_node
        
#         else:
#             new_node = nodes[current_node.val]


#         # Handle the random pointer 
#         if current_node.random is None:
#             new_node.random = None

#         else:

#             # If the randoms does not exist already in the dict, create a new entry in the dict with the random value as key and a node holding that value 
#             if current_node.random.val not in nodes.keys():
#                 nodes[current_node.random.val] = Node(x = current_node.random.val)
          
#             new_node.random = nodes[current_node.random.val]


#         # Move to the next node
#         current_node = current_node.next
    

#     # Pull the nodes as a list to link to their next attribute
#     nodes_list = [nodes[x] for x in old_nodes_vals]

#     # Traverse the nodes list
#     for i, node in enumerate(nodes_list):

#         node.next = nodes_list[i+1] if i != len(nodes_list)-1 else None
   

#     return nodes_list[0]


# result = copyRandomList(head=head)


# new_copy = []
# while result is not None:
#     new_copy.append([result.val, result.random.val if result.random is not None else None])
#     result = result.next


'My solution works while the values of the list are unique, otherwise a new approach is needed'


# Another Approach

# def copyRandomList(head:Node):

#     nodes_map = {}

#     current = head

#     while current is not None:

#         nodes_map[current] = Node(x = current.val)
#         current = current.next

    
#     current = head

#     while current is not None:

#         new_node = nodes_map[current]
#         new_node.next = nodes_map.get(current.next)
#         new_node.random = nodes_map.get(current.random)

#         current = current.next
    
#     return nodes_map[head]


# result = copyRandomList(head=head)


# new_copy = []
# while result is not None:
#     new_copy.append([result.val, result.random.val if result.random is not None else None])
#     result = result.next

'Done'




'''139. Word Break'''

#Input

# #Case 1
# s = "leetcode" 
# wordDict = ["leet","code"]
# #Output: True

# #Case 2
# s = "applepenapple"
# wordDict = ["apple","pen"]
# #Output: True

# #Case 3
# s = "catsandog"
# wordDict = ["cats","dog","sand","and","cat"]
# #Output: False


# My Approach

'''
Intuition:
    (Brute-force): in a while loop go word for word in the dict checking if the 
        word exists in the string:

            - If it does: Overwrite the string taking out the found word / else: go to the next word

        The loop will be when either no words are found in the string or the string is empty

        if after the loop the string is empty, return True, otherwise False
'''

# def workBreak(string:str, word_dict:list[str]) -> bool:

#     j = 0
#     while j < len(word_dict):

#         if word_dict[j] in string:

#             w_len = len(word_dict[j])
#             idx = string.find(word_dict[j])

#             string = string[:idx]+string[idx+w_len:]

#             j = 0
        
#         else:
#             j += 1
    
    
#     return False if string else True

# print(workBreak(string=s, word_dict=wordDict))

'This solution goes up to the 74% of the test cases'

# Dynamic Programming Approach

# def workBreak(string:str, word_dict:list[str]) -> bool:

#     dp = [False] * (len(s) + 1) # dp[i] means s[:i+1] can be segmented into words in the wordDicts 
#     dp[0] = True

#     for i in range(len(s)):

#         for j in range(i, len(s)):
            
#             i_dp = dp[i]
#             sub_s = s[i: j+1]
#             test = sub_s in wordDict

#             if i_dp and test:
#                 dp[j+1] = True
                
#     return dp[-1]

# print(workBreak(string=s, word_dict=wordDict))

'Done'




'''140. Word Break II'''

#Input

# #Case 1
# s = "catsanddog"
# wordDict = ["cat","cats","and","sand","dog"]
# #Output: ["cats and dog","cat sand dog"]

# #Case 2
# s = "pineapplepenapple"
# wordDict = ["apple","pen","applepen","pine","pineapple"]
# #Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]

# #Case 3
# s = "catsandog"
# wordDict = ["cats","dog","sand","and","cat"]
# #Output: []


# My Approach

'''
Intuition:

    - With the solution of the last exercise, bring the found words into a list and join them to from a sentence.
    - In a loop, check if the first found word is the same of the last sentece, if do, keep searching for another word,
        - if not found words after looping from the first character, end the loop.
'''

# def wordBreak(s:str, wordDict:list[str]) -> list[str]:

#     sentences = []
#     sent = []
#     string = s
#     lasts_first_word = []

#     while True:

#         j = 0

#         while j < len(string):

#             if string[0:j+1] in wordDict and string[0:j+1] not in lasts_first_word:

#                 sent.append(string[0:j+1])
#                 string = string[j+1:]
#                 j = 0
            
#             else:
#                 j += 1
        

#         if sent:
#             sentences.append(' '.join(sent))
#             string = s
#             lasts_first_word.append(sent[0])
#             sent = []
        
#         else:
#             break
    
#     return sentences        

# print(wordBreak(s=s, wordDict=wordDict))

"This solution doesn't even get to pass all the initial test cases, but at least it worked as a challenge to do at least one"


# Backtracking & Recursion approach

# def wordBreakHelper(s:str, start:int, word_set:set, memo:dict) -> list[str]:

#     if start in memo:
#         return memo[start]
    
#     valid_substr = []

#     if start == len(s):
#         valid_substr.append('')

#     for end in range(start+1, len(s)+1):

#         prefix = s[start:end]

#         if prefix in word_set:

#             suffixes = wordBreakHelper(s, end, word_set, memo)

#             for suffix in suffixes:

#                 valid_substr.append(prefix + ('' if suffix == '' else ' ') + suffix)

#     memo[start] = valid_substr

#     return valid_substr
         

# def wordBreak(s:str, wordDict: list[str]) -> list[str]:

#     memo = {}
#     word_set = set(wordDict)
#     return wordBreakHelper(s, 0, word_set, memo)


# print(wordBreak(s=s, wordDict=wordDict))

'Done'



'''141. Linked List Cycle'''

# # Base
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


# Input

# # Case 1
# head_layout = [3,2,0,-4]

# head = ListNode(x=3)
# pos1 = ListNode(x=2)
# pos2 = ListNode(x=0)
# pos3 = ListNode(x=-4)

# head.next, pos1.next, pos2.next, pos3.next = pos1, pos2, pos3, pos1
# # Output: True / Pos1

# # Case 2
# head_layout = [1,2]

# head = ListNode(x=1)
# pos1 = ListNode(x=2)

# head.next, pos1.next = pos1, head
# # Output: True / Pos0

# # Case 3
# head_layout = [1]

# head = ListNode(x=1)
# # Output: False / pos-1


# def hasCycle(head:ListNode) -> bool:

#     if head is None or head.next == None:
#         return False
    

#     visited = []

#     curr = head

#     while curr is not None:

#         if curr in visited:
#             return True
        
#         visited.append(curr)

#         curr = curr.next
    
#     return False

# print(hasCycle(head=head))


'This a suboptimal solution, it works but it takes considerable memory to solve it'

# Another approach (Probing)

'''
Explanation
    By making two markers initialized in the head one with the double of the "speed" of the other, if those are in a cycle
    at some point they got to meet, it means there is a cycle in the list, but if one if the faster gets to None,
    that'll mean that there is no cycle in there.
'''

# def hasCycle(head:ListNode) -> bool:

#     if not head:
#         return False
    
#     slow = fast = head

#     while fast and fast.next:

#         slow = slow.next
#         fast = fast.next.next

#         if slow == fast:
#             return True
    
#     return False

# print(hasCycle(head=head))

'Done'




'''146. LRU Cache'''

# Input
# commands = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
# inputs = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

# Output: [null, null, null, 1, null, -1, null, -1, 3, 4]


# My Approach

'''
Intuition

    The use of 'OrderedDicts' from the Collections module will be useful to keep track
    of the last recently used values
'''


# class LRUCache(object):   

#     def __init__(self, capacity):
#         """
#         :type capacity: int
#         """     

#         self.capacity = capacity
#         self.capacity_count = 0
#         self.memory = {}
        

#     def get(self, key):
#         """
#         :type key: int
#         :rtype: int
#         """

#         output = self.memory.get(key,-1)

#         if output != -1:

#             item = (key, self.memory[key])
#             del self.memory[item[0]]
#             self.memory[item[0]] = item[1]

#         return output
        

#     def put(self, key, value):
#         """
#         :type key: int
#         :type value: int
#         :rtype: None
#         """

#         existing_key = self.memory.get(key, -1)

#         if existing_key == -1:
#             self.memory[key] = value

#         else:
#             self.memory.update({key:value})

#             item = (key, value)
#             del self.memory[item[0]]
#             self.memory[item[0]] = item[1]
        
#         self.capacity_count += 1

#         if self.capacity_count > self.capacity:

#             del_item = list(self.memory.keys())[0]
#             del self.memory[del_item]
            
#             self.capacity_count = self.capacity


# a = {'a':1, 'b':2, 'c':3}

# print(a)

# item = ('a', a['a'])

# del a[item[0]]

# a[item[0]] = item[1]


# print(a)

'Done'



'''148. Sort List'''

# Base
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# Input

# # Case 1
# list_layout = [4,2,1,3]
# head = ListNode(val=4, next=ListNode(val=2, next=ListNode(val=1, next=ListNode(val=3))))
# # Output: [1,2,3,4]

# # Case 2
# list_layout = [-1,5,3,4,0]
# head = ListNode(val=-1, next=ListNode(val=5, next=ListNode(val=3, next=ListNode(val=4, next=ListNode(val=0)))))
# # Output: [-1,0,3,4,5]

# # Case 3
# list_layout = [1,2,3,4]
# head = ListNode(val=1, next=ListNode(val=2, next=ListNode(val=3, next=ListNode(val=4))))
# # Output: [1,2,3,4]


# My Approach

'''
Intuition

    - Brute force: Traverse the list to collect each node with its value in a list,
    and apply some sorting algorithm to sort them.

'''

# def sortList(head):

#     if not head:
#         return ListNode()
    
#     curr = head
#     holder = []

#     while curr:

#         holder.append([curr.val, curr])
#         curr = curr.next


#     def merge_sort(li):

#         if len(li)<=1:
#             return li
        
#         left_side = li[:len(li)//2]
#         right_side = li[len(li)//2:]

#         left_side = merge_sort(left_side)
#         right_side = merge_sort(right_side)

#         return merge(left=left_side, right=right_side)


#     def merge(left, right):
        
#         i = j = 0
#         result = []

#         while i < len(left) and j < len(right):

#             if left[i][0] < right[j][0]:
#                 result.append(left[i])
#                 i+=1
            
#             else:
#                 result.append(right[j])
#                 j+=1

#         while i < len(left):
#             result.append(left[i])
#             i+=1
        
#         while j < len(right):
#             result.append(right[j])
#             j+=1

#         return result

#     sorted_list = merge_sort(li=holder)
    
#     for i in range(len(sorted_list)):

#         if i == len(sorted_list)-1:
#             sorted_list[i][1].next = None
        
#         else:
#             sorted_list[i][1].next = sorted_list[i+1][1]
    
#     return sorted_list[0][1]

# test = sortList(head=head)

'Done'




'''149. Max Points on a Line'''

'''
Revision

    The problem could be pretty hard if no math knowledge is acquired beforehand.
    By definition, if several points share the same 'slope' with one single point,
    it'd mean that they are all included in the same line.

    So the problem reduces to (brut force) check for each point if the rest share the same
    slope and the biggest group with common slope will be the answer
'''

# def maxPoints(points:list[list[int]]):

#     # if there is no more than a pair of point in the plane, well, that's the answer
#     if len(points) < 3:
#         return len(points)
    
#     # Initializing with the lowest possible answer
#     result = 2

#     # Since we are counting on pairs, we're iterating up to the second last point of the group
#     for i, point1 in enumerate(points[:-1]):

#         slopes = {} # The keys will be the slopes and the values the count of points with the same slope

#         for point2 in points[i+1:]:
            
#             slope = None
#             x_comp = point2[0] - point1[0]

#             if x_comp:  # The bool of 0 is False
                
#                 # Calculate the slope
#                 slope = (point2[1] - point1[1]) / x_comp

#             # If that slope already exist, add one point to the count
#             if slope in slopes:

#                 slopes[slope] += 1
#                 new = slopes[slope]

#                 result = max(result, new)
            
#             # else, create a new dict entry
#             else:
#                 slopes[slope] = 2

#     return result

'Done'




'''150. Evaluate Reverse Polish Notation'''

# Input

# # Case1
# tokens = ["2","1","+","3","*"]
# #Output: 9 / ((2 + 1) * 3)

# # Case2
# tokens = ["4","13","5","/","+"]
# #Output: 6 / (4 + (13 / 5) )

# # Case3
# tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
# #Output: ((((((9 + 3) * -11) / 6) * 10) + 17) + 5)


# My approach

'''
Intuition

    Look from left to right the first operand and operate with the trailing two elements (which should be digits)
    and return the result to where the first of the digits were and iterate the same process until ther is only
    3 elements in the list to make the last operation

'''

# def evalPRN(tokens:list[str]) -> int:    

#     operations = [x for x in tokens if x in '+-*/']

#     for _ in range(len(operations)):

#         operation = operations.pop(0)
#         idx = tokens.index(operation)
#         return_idx = idx-2

#         operand, num2, num1 = tokens.pop(idx), tokens.pop(idx-1), tokens.pop(idx-2)

#         if operand == '/':

#             num1, num2 = int(num1), int(num2)
#             op_result = num1//num2 if num2 > 0 else (num1 + (-num1 % num2)) // num2

#         else:
#             op_result = eval(num1+operand+num2)

#         tokens.insert(return_idx, str(op_result))
        
#     return tokens[0]


# print(evalPRN(tokens=tokens))

'My solution worked for 81% of the cases'

# # Another Approach

# def evalRPN(tokens):
        
#     stack = []

#     for x in tokens:

#         if stack == []:
#             stack.append(int(x))

#         elif x not in '+-/*':
#             stack.append(int(x))

#         else:

#             l = len(stack) - 2

#             if x == '+':
#                 stack[l] = stack[l] + stack.pop()

#             elif x == '-':
#                 stack[l] = stack[l] - stack.pop()

#             elif x == '*':
#                 stack[l] = stack[l] * stack.pop()

#             else:
#                 stack[l] = float(stack[l]) / float(stack.pop())
#                 stack[l] = int(stack[l])    

#     return stack[0]

# print(evalRPN(tokens=tokens))

'Done'




'''152. Maximum Product Subarray'''

# Input

# # Case 1
# input = [2,3,-2,4]
# # Output: 6 / [2,3] has the largest product

# # Case 2
# input = [-2,0,-1]
# # Output: 0 / all products are 0

# # Custom Case
# input = [-2,3,-4]
# # Output: 0 / all products are 0


# My approach

'''
Intuition

    This is a variation of Kadane's Algorithm, and may be solve same way
    as the original
'''

# def maxProduct(nums:list[int]) -> int:

#     if len(nums) == 1:
#         return nums[0]

#     max_ends_here, max_so_far = nums[0]

#     for num in nums[1:]:
       
#         max_ends_here = max(num, max_ends_here * num)
#         max_so_far = max(max_so_far, max_ends_here)

#     return max_so_far

# print(maxProduct(nums=input))

'''
Original Kadane's modified to compute product solved 51% of the cases. 
But, apparently with capturing the min_so_far and having a buffer to hold the max_so_far to not interfere with the
    min_so_far calculation, the problem is solved
'''

# Another Kadane's Mod. Approach

# def maxProduct(nums:list[int]) -> int:

#     if len(nums) == 1:
#         return nums[0]

#     max_so_far = min_so_far = result = nums[0]

#     for num in nums[1:]:
       
#         temp_max = max(num, max_so_far * num, min_so_far * num)
#         min_so_far = min(num, max_so_far * num, min_so_far * num)
#         max_so_far = temp_max

#         result = max(result, max_so_far)

#     return result

# print(maxProduct(nums=input))

'Done'




'''155. Min Stack'''

# Input

# # Case 1
# commands = ["MinStack","push","push","push","getMin","pop","top","getMin"]
# inputs = [[],[-2],[0],[-3],[],[],[],[]]
# # Output: [None,None,None,None,-3,None,0,-2]

# # Custom Case
# commands = ["MinStack","push","push","push","top","pop","getMin","pop","getMin","pop","push","top","getMin","push","top","getMin","pop","getMin"]
# inputs = [[],[2147483646],[2147483646],[2147483647],[],[],[],[],[],[],[2147483647],[],[],[-2147483648],[],[],[],[]]
# # Output: [None,None,None,None,-3,None,0,-2]


# Solution

# class MinStack(object):

#     def __init__(self):
#         self.stack = []
#         self.min = None
        

#     def push(self, val):
#         """
#         :type val: int
#         :rtype: None
#         """
#         self.stack.append(val)

#         if not self.min:
#             self.min = val
        
#         else:
#             self.min = min(val, self.min)
        

#     def pop(self):
#         """
#         :rtype: None
#         """
#         item = self.stack.pop()

#         if item == self.min:
            
#             self.min = min(self.stack) if self.stack else None


#     def top(self):
#         """
#         :rtype: int
#         """
#         return self.stack[-1]
        

#     def getMin(self):
#         """
#         :rtype: int
#         """
#         return self.min
        

# # For testing
# for i, command in enumerate(commands):

#     if command == 'MinStack':
#         stack = MinStack()
    
#     elif command == 'push':
#         stack.push(inputs[i][0])   

#     elif command == 'pop':
#         stack.pop()    
    
#     elif command == 'top':
#         res = stack.top()

#     elif command == 'getMin':
#         res = stack.getMin()

'My solution worked for 97% of the cases'


# Another solution

# class MinStack(object):

#     def __init__(self):
#         self.stack = []
                

#     def push(self, val):
#         """
#         :type val: int
#         :rtype: None
#         """

#         if not self.stack:
#             self.stack.append([val, val])
#             return
        
#         min_elem = self.stack[-1][1]

#         self.stack.append([val, min(val, min_elem)])
        

#     def pop(self):
#         """
#         :rtype: None
#         """
#         self.stack.pop()
        

#     def top(self):
#         """
#         :rtype: int
#         """
#         return self.stack[-1][0]
        

#     def getMin(self):
#         """
#         :rtype: int
#         """
#         return self.stack[-1][1]

'Done'




'''160. Intersection of Two Linked Lists'''

# # Base

# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Input

# # Case 1
# listA, listB = [4,1,8,4,5], [5,6,1,8,4,5]

# a1, a2 = ListNode(x=4), ListNode(x=1)
# c1, c2, c3 = ListNode(x=8), ListNode(x=4), ListNode(x=5)
# b1, b2, b3 = ListNode(x=5), ListNode(x=6), ListNode(x=1)

# a1.next, a2.next = a2, c1
# c1.next, c2.next = c2, c3
# b1.next, b2.next, b3.next = b2, b3, c1
# #Output: 8

# # Case 2
# listA, listB = [1,9,1,2,4], [3,2,4]

# a1, a2, a3 = ListNode(x=1), ListNode(x=9), ListNode(x=1)
# c1, c2 = ListNode(x=2), ListNode(x=4)
# b1 = ListNode(x=3)

# a1.next, a2.next, a3.next = a2, a3, c1
# c1.next = c2
# b1.next = c1
# # Output: 2

# # Case 3
# listA, listB = [2,6,4], [1,5]

# a1, a2, a3 = ListNode(x=2), ListNode(x=6), ListNode(x=4)

# b1, b2 = ListNode(x=1), ListNode(x=5)

# a1.next, a2.next = a2, a3
# b1.next = b2
# # Output: None


# My approach

'''
Intuition
    - Traverse the first list saving the nodes in a list
    - Traverse the second list while checking if the current node is in the list
        - If so, return that node
        - Else, let the loop end
    - If the code gets to the end of the second loop, means there isn't a intersection.
'''

# def getIntersectionNode(headA = ListNode, headB = ListNode) -> ListNode:

#     visited_nodes = []

#     curr = headA

#     while curr:
#         visited_nodes.append(curr)
#         curr = curr.next

#     curr = headB

#     while curr:
        
#         if curr in visited_nodes:
#             return curr
        
#         curr = curr.next
        
#     return None


# result = getIntersectionNode(headA=a1, headB=b1)

# print(result.val) if result else print(None)

'This solution breaks when the data input is too large in leetcode, it got up to 92% of cases'

# # Two pointers Approach

# def getIntersectionNode(headA = ListNode, headB = ListNode) -> ListNode:

#     a, b = headA, headB

#     while a != b:
       
#         if not a:
#             a = headB

#         else:
#             a = a.next
        
#         if not b:
#             b = headA
        
#         else:
#             b = b.next
    
#     return a


# result = getIntersectionNode(headA=a1, headB=b1)

# print(result.val) if result else print(None)


'''
Explanation

    The logic here is that with two pointer, each one directed to the head of each list,
    if both exhaust their lists and star with the other, if there are intersected they MUST
    meet at the intersection node after traversing both lists respectviely or otherwise they will be 'None'
    at same time after the second lap of the respective lists.
'''




'''xxx'''










