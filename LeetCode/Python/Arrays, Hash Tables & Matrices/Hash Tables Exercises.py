'''
CHALLENGES INDEX

3. Longest Substring Without Repeating Characters (Hash Table) (SW)
13. Roman to Integer (Hash Table)
17. Letter Combinations of a Phone Number (Hash Table) (BT)
73. Set Matrix Zeroes (Matrix) (Hash Table)
76. Minimum Window Substring (Hash Table) (SW)
127. Word Ladder (Hast Table) (BFS)
138. Copy List with Random Pointer (Hash Table) (LL)
146. LRU Cache (Hash Table)
166. Fraction to Recurring Decimal (Hash Table) (Others)
202. Happy Number (Hash Table) (TP) (Others)
208. Implement Trie (Hast Table) (Tree)
380. Insert Delete GetRandom O(1) (Hash Table) (Others)

142. Linked List Cycle II (Hash Table) (LL) (TP) (FCD)



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


(13)
'''


'3. Longest Substring Without Repeating Characters'
# def x():
#     s = "abcabcbb"


#     # My solution
#     substrings = []

#     i = 0

#     while i < len(s):

#         sub = str()

#         for char in s[i:]:

#             if char in sub:
#                 substrings.append(sub)
#                 break

#             sub += char
        
#         if sub not in substrings:
#             substrings.append(sub)

#         i += 1

#     # print(substrings)

#     max_sub = max(substrings, key = len) if substrings else 0

#     # print(max_sub)

#     print(max_sub, len(max_sub))


#     # Another more efficient solution

#     def lengthOfLongestSubstring(s: str) -> int:
            
#             n = len(s)
#             maxLength = 0
#             charMap = {}
#             left = 0
            
#             for right in range(n):

#                 if s[right] not in charMap or charMap[s[right]] < left:
#                     charMap[s[right]] = right
#                     maxLength = max(maxLength, right - left + 1)

#                 else:
#                     left = charMap[s[right]] + 1
#                     charMap[s[right]] = right
            
#             return maxLength


#     lengthOfLongestSubstring(s)

'13. Roman to Integer'
# def x():

#     '''
#     Substraction exceptions:
#         - I can be placed before V (5) and X (10) to make 4 and 9. 
#         - X can be placed before L (50) and C (100) to make 40 and 90. 
#         - C can be placed before D (500) and M (1000) to make 400 and 900.
#     '''

#     # Input
#     s = 'MCMXCIV'


#     # My approach
#     res = 0
#     rom_to_int_dic = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, 'CM': 900, 'M': 1000, }


#     #Substraction Exceptions
#     if 'IV' in s:
#         res += rom_to_int_dic['IV']
#         s = s.replace('IV','')

#     if 'IX' in s:
#         res += rom_to_int_dic['IX']
#         s = s.replace('IX','')

#     if 'XL' in s:
#         res += rom_to_int_dic['XL']
#         s = s.replace('XL','')

#     if 'XC' in s:
#         res += rom_to_int_dic['XC']
#         s = s.replace('XC','')

#     if 'CD' in s:
#         res += rom_to_int_dic['CD']
#         s = s.replace('CD','')

#     if 'CM' in s:
#         res += rom_to_int_dic['CM']
#         s = s.replace('CM','')

#     # Dealing with the Remaining Number
#     if s:
#         for chr in s:
#             res += rom_to_int_dic[chr]

#     else:
#         print(res)


#     print(res)

#     '''
#     Note: This version works, but there is a more concise way
#     '''

#     s = 'MCMXCIV'

#     # ChatGPT's Approach
#     roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
#     total = 0
#     prev_value = 0

#     for char in s[::-1]:    #Reverse to simplify the process
        
#         curr_value = roman_dict[char]

#         if curr_value < prev_value:
#             total -= curr_value
        
#         else:
#             total += curr_value
#             prev_value = curr_value

#     print(total)

'17. Letter Combinations of a Phone Number'
# def x():

#     # Input
#     s = '23'

#     '''
#     My Approach

#     Rationale: Brute-forcing 
#         Iterating by the number of characters there are.
#     '''

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }


#         lists = []

#         for i in digits:
#             lists.append(dic[i])

        
#         comb = []

#         if not digits:
#             return comb


#         if len(digits) == 4:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     for k in range(len(lists[2])):
#                         for l in range(len(lists[3])):
#                             # comb.append(f'{lists[0][i]}{lists[1][j]}{lists[2][k]}{lists[3][l]}')
#                             comb.append(lists[0][i]+lists[1][j]+lists[2][k]+lists[3][l])
    
#         elif len(digits) == 3:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     for k in range(len(lists[2])):
#                         comb.append(f'{lists[0][i]}{lists[1][j]}{lists[2][k]}')

#         elif len(digits) == 2:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     comb.append(f'{lists[0][i]}{lists[1][j]}')

#         elif len(digits) == 1:
#             for i in range(len(lists[0])):
#                 comb.append(f'{lists[0][i]}')


#         return comb


#     print(letterCombinations(s))

#     'Notes: It works but it could be better'


#     # Recursive Approach

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }

#         lists = []

#         for i in digits:
#             lists.append(dic[i])


#         def combine_sublists(lst):

#             if len(lst) == 1:
#                 return lst[0]
            
#             result = []

#             for item in lst[0]:
#                 for rest in combine_sublists(lst[1:]):
#                     result.append(item + rest)
            
#             return result
        
#         if not digits:
#             return []
        
#         else:
#             return combine_sublists(lists)

#     print(letterCombinations(s))

#     'Notes: Works pretty well'


#     # Itertools Approach

#     import itertools

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }

#         lists = []

#         for i in digits:
#             lists.append(dic[i])

#         if not digits:
#             return []
        
#         else:
#             return [''.join(comb) for comb in itertools.product(*lists)]
        

#     print(letterCombinations(s))

'73. Set Matrix Zeroes'
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,1,1],[1,0,1],[1,1,1]]
#     # Output: [[1,0,1],[0,0,0],[1,0,1]]

#     # Case 2
#     matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
#     # Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

#     # Custom Case
#     matrix = [...]
#     # Output: [...]

#     '''
#     My Approach

#         Intuition:
#             - Locate the indexes of every 0 present
#             - Try to overwrite the values for the row and the column of each occurrence
#             - Look up if the col and row are already 0 to optimize
#     '''

#     def setZeroes(matrix: list[list[int]]) -> list[list[int]]:

#         m, n = len(matrix), len(matrix[0])
#         occurrences = []

#         for i, row in enumerate(matrix):

#             for j, col in enumerate(row):

#                 if 0 not in row:
#                     continue
                
#                 if col == 0:
#                     occurrences.append((i,j))

#         for pair in occurrences:

#             matrix[pair[0]] = [0] * n

#             for row in range(m):
#                 matrix[row][pair[1]] = 0
        
#         return matrix

#     # Testing
#     for i in setZeroes(matrix):
#         print(i)

#     '''
#     Notes: It actually passed! :D
#     '''

'76. Minimum Window Substring'
# def x():

#     # Input
#     # Case 1
#     s, t = 'ADOBECODEBANC', 'ABC'
#     # Output: "BANC"

#     # Case 2
#     s, t = 'a', 'a'
#     # Output: "a"

#     # Case 3
#     s, t = 'a', 'aa'
#     # Output: "abbbbbcdd"

#     # Custom case
#     s, t = 'aaaaaaaaaaaabbbbbcdd', 'abcdd'
#     # Output: "abbbbbcdd"


#     'My approach'
#     def minWindow(s:str, t:str) -> str:

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 if all([char in s[j:j+i] for char in t]):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 57%'


#     'With an improvement'
#     def minWindow(s:str, t:str) -> str:

#         from collections import Counter

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        
#         count_t = Counter(t).items()

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 subs = s[j:j+i]
#                 count_subs = Counter(subs)

#                 if all( (x[0] in count_subs.keys() and x[1] <= count_subs[x[0]]) for x in count_t):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 93% and hit the time limit'


#     'Another solution'
#     def minWindow(s, t):    

#         if not s or not t:
#             return ""


#         from collections import defaultdict

#         dictT = defaultdict(int)
#         for c in t:
#             dictT[c] += 1

#         required = len(dictT)
#         l, r = 0, 0
#         formed = 0

#         windowCounts = defaultdict(int)
#         ans = [-1, 0, 0]

#         while r < len(s):
#             c = s[r]
#             windowCounts[c] += 1

#             if c in dictT and windowCounts[c] == dictT[c]:
#                 formed += 1

#             while l <= r and formed == required:
#                 c = s[l]

#                 if ans[0] == -1 or r - l + 1 < ans[0]:
#                     ans[0] = r - l + 1
#                     ans[1] = l
#                     ans[2] = r

#                 windowCounts[c] -= 1
#                 if c in dictT and windowCounts[c] < dictT[c]:
#                     formed -= 1

#                 l += 1

#             r += 1

#         return "" if ans[0] == -1 else s[ans[1]:ans[2] + 1]
            
#     # Testing
#     print(minWindow(s,t))

'''127. Word Ladder'''
# def x():

#     # Input
#     #Case 1
#     begin_word, end_word, word_list = 'hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']
#     #Output: 5

#     #Custom Case
#     begin_word, end_word, word_list = 'a', 'c', ['a', 'b', 'c']
#     #Output: 5
    

#     '''
#     My Approach

#         Intuition:
#             1. handle the corner case: the end_word not in the word_list
#             2. create an auxiliary func that check the word against the end_word: True if differ at most by 1 char, else False.
#             3. create a counter initialized in 0
#             4. start checking the begin_word and the end_word, if False sum 1 to the count, and change to the subquent word in the word_list and do the same.
#     '''

#     def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#         if endWord not in wordList:
#             return 0
        
#         def check(word):
#             return False if len([x for x in word if x not in endWord]) > 1 else True
        
#         if beginWord not in wordList:
#             wordList.insert(0,beginWord)
#             count = 0
        
#         else:
#             count = 1
        
#         for elem in wordList:
#             count += 1

#             if check(elem):
#                 return count     
                
#         return 0

#     # Testing
#     print(ladderLength(beginWord=begin_word, endWord=end_word, wordList=word_list))

#     'Note: This solution only went up to the 21% of the cases'


#     'BFS approach'
#     from collections import defaultdict, deque

#     def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#         if endWord not in wordList or not endWord or not beginWord or not wordList:
#             return 0

#         L = len(beginWord)
#         all_combo_dict = defaultdict(list)

#         for word in wordList:
#             for i in range(L):
#                 all_combo_dict[word[:i] + "*" + word[i+1:]].append(word) 

#         queue = deque([(beginWord, 1)])
#         visited = set()
#         visited.add(beginWord)

#         while queue:
#             current_word, level = queue.popleft()

#             for i in range(L):
#                 intermediate_word = current_word[:i] + "*" + current_word[i+1:]

#                 for word in all_combo_dict[intermediate_word]:

#                     if word == endWord:
#                         return level + 1

#                     if word not in visited:
#                         visited.add(word)
#                         queue.append((word, level + 1))
                        
#         return 0

#     'Done'

'''138. Copy List with Random Pointer'''
# def x():

#     # Base
#     class Node:
#         def __init__(self, x, next=None, random=None):
#             self.val = int(x)
#             self.next = next
#             self.random = random


#     #Input
#     #Case 1
#     head_map = [[7,None],[13,0],[11,4],[10,2],[1,0]]

#     #Build the relations of the list
#     nodes = [Node(x=val[0]) for val in head_map]

#     for i in range(len(nodes)):
#         nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#         nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

#     head = nodes[0]
#     # Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


#     #Case 2
#     head_map = [[1,1],[2,1]]

#     #Build the relations of the list
#     nodes = [Node(x=val[0]) for val in head_map]

#     for i in range(len(nodes)):
#         nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#         nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

#     head = nodes[0]
#     #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


#     #Case 3
#     head_map = [[3,None],[3,0],[3,None]]

#     #Build the relations of the list
#     nodes = [Node(x=val[0]) for val in head_map]

#     for i in range(len(nodes)):
#         nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#         nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

#     head = nodes[0]
#     #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


#     '''
#     My Approach

#         Intuition:
#             - Traverse through the list
#             - Create a copy of each node and store it into a list along side with the content of the random pointer.
#             - Traverse the list linking each node to the next and the random pointer to the position in that list.

#         Thoughts:

#         - It is possible to create the list with a recursive solution but it'll be still necesary to traverse again
#             to collect the content of the random pointer or how else I can point to somewhere at each moment I don't know if it exist. 
#     '''

#     def copyRandomList(head:Node) -> Node:

#         # Handle the corner case where there is a single node list
#         if head.next == None:
#             result = Node(x = head.val, random=result)
#             return result

#         # Initilize a nodes holder dict to collect the new nodes while traversing the list
#         nodes = {}

#         # Initilize a nodes holder list to collect the old nodes values while traversing the list
#         old_nodes_vals = []

#         # Initialize a dummy node to traverse the list
#         current_node = head

#         # Traverse the list
#         while current_node is not None:

#             # Collect the old nodes
#             old_nodes_vals.append(current_node.val)

#             # Check if the node doesn't already exist due to the random pointer handling
#             if current_node.val not in nodes.keys(): 

#                 new_node = Node(x = current_node.val)
#                 nodes[new_node.val] = new_node
            
#             else:
#                 new_node = nodes[current_node.val]


#             # Handle the random pointer 
#             if current_node.random is None:
#                 new_node.random = None

#             else:

#                 # If the randoms does not exist already in the dict, create a new entry in the dict with the random value as key and a node holding that value 
#                 if current_node.random.val not in nodes.keys():
#                     nodes[current_node.random.val] = Node(x = current_node.random.val)
            
#                 new_node.random = nodes[current_node.random.val]


#             # Move to the next node
#             current_node = current_node.next
        

#         # Pull the nodes as a list to link to their next attribute
#         nodes_list = [nodes[x] for x in old_nodes_vals]

#         # Traverse the nodes list
#         for i, node in enumerate(nodes_list):
#             node.next = nodes_list[i+1] if i != len(nodes_list)-1 else None    

#         return nodes_list[0]

#     # Testing
#     result = copyRandomList(head=head)

#     new_copy = []
#     while result is not None:
#         new_copy.append([result.val, result.random.val if result.random is not None else None])
#         result = result.next

#     'Note: My solution works while the values of the list are unique, otherwise a new approach is needed'


#     'Another Approach'
#     def copyRandomList(head:Node):

#         nodes_map = {}

#         current = head

#         while current is not None:

#             nodes_map[current] = Node(x = current.val)
#             current = current.next

        
#         current = head

#         while current is not None:

#             new_node = nodes_map[current]
#             new_node.next = nodes_map.get(current.next)
#             new_node.random = nodes_map.get(current.random)

#             current = current.next
        
#         return nodes_map[head]


#     result = copyRandomList(head=head)


#     new_copy = []
#     while result is not None:
#         new_copy.append([result.val, result.random.val if result.random is not None else None])
#         result = result.next

#     'Done'

'''146. LRU Cache'''
# def x():

#     # Input
#     commands = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
#     inputs = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
#     # Output: [null, null, null, 1, null, -1, null, -1, 3, 4]


#     '''
#     My Approach
    
#         Intuition:

#             - The use of 'OrderedDicts' from the Collections module will be useful to keep track of the last recently used values
#     '''

#     class LRUCache(object):   

#         def __init__(self, capacity):
#             """
#             :type capacity: int
#             """     

#             self.capacity = capacity
#             self.capacity_count = 0
#             self.memory = {}
            

#         def get(self, key):
#             """
#             :type key: int
#             :rtype: int
#             """

#             output = self.memory.get(key,-1)

#             if output != -1:

#                 item = (key, self.memory[key])
#                 del self.memory[item[0]]
#                 self.memory[item[0]] = item[1]

#             return output
            

#         def put(self, key, value):
#             """
#             :type key: int
#             :type value: int
#             :rtype: None
#             """

#             existing_key = self.memory.get(key, -1)

#             if existing_key == -1:
#                 self.memory[key] = value

#             else:
#                 self.memory.update({key:value})

#                 item = (key, value)
#                 del self.memory[item[0]]
#                 self.memory[item[0]] = item[1]
            
#             self.capacity_count += 1

#             if self.capacity_count > self.capacity:

#                 del_item = list(self.memory.keys())[0]
#                 del self.memory[del_item]
                
#                 self.capacity_count = self.capacity

#     'Done'

'''166. Fraction to Recurring Decimal'''
# def x():

#     # Input
#     # Case 1
#     num, den = 1, 2
#     # Output: "0.5"

#     # Case 2
#     num, den = 2, 1
#     # Output: "2"

#     # Case 3
#     num, den = 4, 333
#     # Output: "0.(012)"

#     # Custom Case 
#     num, den = 1, 6
#     # Output: "0.1(6)"


#     '''
#     My approach
        
#         Intuition:

#             Here main issue is solving how to identify patterns in a string:
#                 - I'll try with parsing the string with split()
#     '''

#     def fraction_to_decimal(numerator: int, denominator: int) -> str:

#         # If exact division
#         if int(numerator/denominator) == numerator/denominator:
#             return str(int(numerator/denominator))
        
#         division = str(numerator/denominator)

#         whole, decimal = division.split('.')

#         pattern = ''

#         for i in range(len(decimal)-1):

#             pattern += decimal[i]
#             abr = decimal.split(pattern)

#             if not any(abr):
#                 return f'{whole}.({pattern})'            
        
#         return f'{whole}.{decimal}'

#     # Testing
#     print(fraction_to_decimal(numerator=num, denominator=den))

#     '''Note: My solution only solved 50% of the cases because it only works if the whole decimal part is recurring and also didnt considered negatives results'''


#     'Hashmap / Long division Approach'
#     def fraction_to_decimal(numerator: int, denominator: int) -> str:

#         # If exact division
#         if numerator % denominator == 0:
#             return str(numerator//denominator)
        
#         # Determe if is a negative result
#         sign = '-' if numerator * denominator < 0 else None

#         # Work with absolutes to simplify the calculation
#         numerator, denominator = abs(numerator), abs(denominator)

#         # Initialize integer and decimal parts
#         integer_part = numerator // denominator
#         remainder = numerator % denominator

#         decimal_part = ''
#         remainder_dict = {}

#         # Track the position of the decimals
#         position = 0

#         # Build the decimal part
#         while remainder != 0:

#             if remainder in remainder_dict:

#                 repeat_start = remainder_dict[remainder]
#                 non_repeaing_part = decimal_part[:repeat_start]
#                 repeating_part = decimal_part[repeat_start:]
#                 return f'{integer_part}.{non_repeaing_part}({repeating_part})' if not sign else f'-{integer_part}.{non_repeaing_part}({repeating_part})'

#             remainder_dict[remainder] = position
#             remainder *= 10
#             digit = remainder // denominator
#             decimal_part += str(digit)
#             remainder %= denominator
#             position += 1
        
#         return f'{integer_part}.{decimal_part}' if not sign else f'-{integer_part}.{decimal_part}'

#     # Testing
#     print(fraction_to_decimal(numerator=num, denominator=den))

#     '''Note: The final solution were based on understanding how long division works and when to capture the moment when is repeating the remainders'''

'''202. Happy Number'''
# def x():

#     # Input
#     # Case 1
#     n = 19
#     # Output: True

#     # Case 2
#     n = 2
#     # Output: False

#     # Custom Case
#     n = 18
#     # Output: False


#     '''
#     My Approach
    
#         Intuition (Recursive)
            
#             - Recursively separate the digits and check the sum of their squares compared to 1.
#                 - If the stackoverflow is reached, return False.            
#     '''

#     def isHappy(n:int) -> bool:

#         def aux(m:int) -> bool:

#             num = [int(x)**2 for x in str(m)]
#             num = sum(num)

#             if num == 1:
#                 return True
            
#             return aux(m=num)
        
#         try:
#             res = aux(m=n)

#             if res:
#                 return True
        
#         except RecursionError as e:        
#             return False

#     # Testing
#     print(isHappy(n=n))

#     'This approach may work but it exceed time limit: only met 4% of cases'


#     '''
#     Set Approach

#     There are mainly two ways of solving this: The set approach and the Floyd's Cycle detection algorithm

#         - The set approach: Use a set to save the seen numbers and if you end up in one of them, you entered a cycle
#         - The Floyd's Cycle Detection Algorithm: Similar to the case of catching a cycle in a linked list with two pointers: Slow and Fast.
#     '''

#     'Set Approach'
#     def isHappy(n:int) -> bool:

#         def getNum(m:int)->int:
#             return sum(int(x)**2 for x in str(m))

#         seen = set()

#         while n != 1 and n not in seen:
#             seen.add(n)
#             n = getNum(n)
        
#         return n == 1

#     # Testing
#     print(isHappy(n=n))


#     'FDC Approach'
#     def isHappy(n:int) -> bool:

#         def getNum(m:int)->int:
#             return sum(int(x)**2 for x in str(m))

#         slow = n
#         fast = getNum(n)

#         while fast != 1 and slow != fast:
#             slow = getNum(slow)
#             fast = getNum(getNum(fast))
        
#         return fast == 1

#     # Testing
#     print(isHappy(n=n))

#     'Done'

'''208. Implement Trie (Prefix Tree)'''
# def x():

#     # Implementation
#     class TrieNode:

#         def __init__(self, is_word=False):
#             self.values = {}
#             self.is_word = is_word

#     'Solution'
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
    

#         def insert(self, word: str) -> None:

#             node = self.root

#             for char in word:

#                 if char not in node.values:
#                     node.values[char] = TrieNode()
                
#                 node = node.values[char]

#             node.is_word = True


#         def search(self, word: str) -> bool:
            
#             node = self.root

#             for char in word:          
                        
#                 if char not in node.values:
#                     return False
                
#                 node = node.values[char]
            
#             return node.is_word


#         def startsWith(self, prefix: str) -> bool:
            
#             node = self.root

#             for char in prefix:

#                 if char not in node.values:
#                     return False
                
#                 node = node.values[char]
            
#             return True

#     # Testing
#     new_trie = Trie()
#     new_trie.insert('Carrot')
#     print(new_trie.startsWith('Car'))  

#     'Done'

'''380. Insert Delete GetRandom O(1)'''
# def x():

#     # Input
#     # Case 1
#     operations = ["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
#     inputs = [[], [1], [2], [2], [], [1], [2], []]
#     # Output: [None, true, false, true, 2, true, false, 2]


#     'My approach'
#     import random

#     class RandomizedSet:

#         def __init__(self):
#             self.set: set = set()
#             print('Set created!')
            

#         def insert(self, val: int) -> bool:

#             if val not in self.set:
#                 self.set.add(val)
#                 return True

#             else:
#                 return False
            

#         def remove(self, val: int) -> bool:

#             if val in self.set:
#                 self.set.remove(val)
#                 return True
            
#             else:
#                 return False
            

#         def getRandom(self) -> int:

#             return random.choice(list(self.set))

#     'Note: While this approach works, it has O(1) time complexity for all the functions, the list casting in the getRandom() function make it go up to O(n) breaking the challenge requirement'


#     'An optimal solution'
#     import random

#     class RandomizedSet:

#         def __init__(self):
#             self.list = []
#             self.dict = {}
                    
#         def insert(self, val: int) -> bool:

#             if val in self.dict:
#                 return False
            
#             self.dict[val] = len(self.list)
#             self.list.append(val)

#             return True

#         def remove(self, val: int) -> bool:

#             if val not in self.dict:
#                 return False
            
#             last_value, idx = self.list[-1], self.dict[val]

#             # Rewrite the list and the dict
#             self.list[idx], self.dict[last_value] = last_value, idx

#             # Update the list to remove the duplicate
#             self.list.pop()

#             # Remove the value entry in the dict
#             del self.dict[val]

#             return True
            
#         def getRandom(self) -> int:

#             return random.choice(self.list)

#     # Testing
#     for i, op in enumerate(operations):

#         if op == 'RandomizedSet':
#             obj = RandomizedSet()
            
#         elif op == 'insert':
#             print(obj.insert(inputs[i][0]))
        
#         elif op == 'remove':
#             print(obj.remove(inputs[i][0]))
        
#         else:
#             print(obj.getRandom())

#     'Done'




'''142. Linked List Cycle II'''
# def x():
    
#     from typing import Optional

#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, x):
#             self.val = x
#             self.next = None

#     # Input
#     # Case 1
#     head_layout = [3,2,0,-4]
#     head = ListNode(3)
#     two = ListNode(2)
#     three = ListNode(0)
#     four = ListNode(-4)
#     head.next, two.next, three.next, four.next = two, three, four, two
#     # Output: Node in position 1

#     # # Case 2
#     # head_layout = [1,2]
#     # head = ListNode(1)
#     # two = ListNode(2)
#     # head.next, two.next = two, head
#     # # Output: Node in position 0
    
#     # Custom Case
#     head_layout = [1,2]
#     head = ListNode(1)
#     two = ListNode(2)
#     head.next, two.next = None, None
#     # Output: Node in position 0

#     '''
#     My Approach (Hash Table Approach)

#         Intuition:
            
#             - Create a Hash Table that stores each node while traversing the list as key and the node pointed by its 
#                 'next' attribute as its value.

#                 Check Membership of each value to be added, 
                    
#                     if it's already in the hash table, return that value as result.
#                     if the value is None, then return None as no cycle is in that list.
#     '''

#     def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Create the Hash Table holder
#         nodes = {}

#         # Initialize a node in head to traverse the list
#         curr = head

#         while curr:

#             if curr.next in nodes:

#                 # If the node pointed by 'next' is already in the table, that's the begining of the cylce
#                 return curr.next
            
#             # Add the node to the holder
#             nodes[curr] = curr.next

#             # Move up to the next node
#             curr = curr.next
        
#         # If the loop finishes it means there was no cycle
#         return None

#     # Testing
#     print(detectCycle(head=head).val)


#     '''Note: This approach worked but it consumes O(n) memory, and is asked if could be done in O(1). Runtime: 12%; Memory: 8%'''




#     '''
#     My Approach (FCD/Tortoise and Hare Approach)

#         To reach the O(1) Memory
#         Intuition:
            
#             Build the Totroise and Hare cycle detection .
#     '''

#     def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Initialize both pointers to the beginning of the list
#         slow, fast = head, head

#         # Traverse the LList
#         while fast and fast.next:

#             slow = slow.next
#             fast = fast.next.next

#             # If slow and fast meet, a cycle exists
#             if slow == fast:

#                 # Move slow pointer back to head and move both pointers one step at a time
#                 slow = head

#                 while slow != fast:
#                     slow = slow.next
#                     fast = fast.next

#                 # Return the node where they meet, i.e., the cycle's starting point
#                 return slow

#         # If the loop finishes without a meeting point, there is no cycle
#         return None

#     # Testing
#     print(detectCycle(head=head))





















