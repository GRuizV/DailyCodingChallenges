'''
CHALLENGES INDEX

15. 3Sum (Array) (TP) (Sorting)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
49. Group Anagrams (Array) (Hash Table) (Sorting)


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
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs, Trees & Binary Trees

(XX)
'''


'15. 3Sum'
# def x():

#     import itertools

#     # Input
#     nums = [0,0,0]


#     # My approach

#     '''
#     Rationale:
        
#         1) Build all combinations caring for the order.
#         2) Filter down those who met sum(subset) = 0
#         3) Make sure there is no duplicates & return.

#     '''
#     comb = list(itertools.combinations(nums,3))

#     comb = [sorted(x) for x in comb if sum(x) == 0]

#     res = []

#     for i in comb:

#         if i not in res:
#             res.append(i)

#     print(res)

#     '''
#     Notes:

#         This solution actually works, but breaks when a big enough input is passed.
#     '''

#     # Two-Pointers approach solution
#     def threeSum(self, nums):
            
#             nums.sort()
#             answer = []
            
#             # if the inputs have less than 3 items
#             if len(nums) < 3:
#                 return answer
            
#             for i in range(len(nums)):

#                 # Since is a sorted input, if first element is positive, there is no way it'll sum up to 0
#                 if nums[i] > 0:
#                     break
                
#                 # Apart from the first element, if the following is the same, jump to the next iteration to avoid returning duplicates
#                 if i > 0 and nums[i] == nums[i - 1]:
#                     continue
                
#                 # Pointers setting    
#                 low, high = i + 1, len(nums) - 1

#                 while low < high:

#                     s = nums[i] + nums[low] + nums[high]

#                     if s > 0:
#                         high -= 1

#                     elif s < 0:
#                         low += 1

#                     else:

#                         answer.append([nums[i], nums[low], nums[high]])
#                         lastLowOccurrence, lastHighOccurrence = nums[low], nums[high]
                        
#                         while low < high and nums[low] == lastLowOccurrence:
#                             low += 1
                        
#                         while low < high and nums[high] == lastHighOccurrence:
#                             high -= 1
            
#             return answer

'23. Merge k Sorted Lists'
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input

#     # 1st Input
#     #List 1
#     one1, two1, three1 = ListNode(1), ListNode(4), ListNode(5)
#     one1.next, two1.next = two1, three1

#     #List 2
#     one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
#     one2.next, two2.next = two2, three2

#     #List 3
#     one3, two3 = ListNode(2), ListNode(6)
#     one3.next = two3

#     # List of lists
#     li = [one1, one2, one3]

#     # My Approach

#     '''
#     Rationale:
    
#         1. Create an empty node.
#         2. Assign the node with the minimum value as next
#         3. Move that node to its next node until reaches 'None'.
#         4. When every value within the input list is None, breakout the loop and return.
#     '''

#     def mergeKLists(lists:list[ListNode]) -> ListNode:
        
#         lists = [x for x in lists if x.val != '']

#         if len(lists) == 0:
#             return ListNode('')


#         head = ListNode('')
#         curr = head
#         li = lists

#         while True:

#             if li == [None]:
#                 break

#             # Create a list of the current nodes in input that aren't None and sort them ascendingly by value
#             li = sorted([node for node in li if node != None], key = lambda x: x.val)

#             # Make the 'next_node' the next node to the curr None & move over to that node right away
#             curr.next = li[0]
#             curr = curr.next

#             # Move over to the next node of next_node
#             li[0] = li[0].next

#         return head.next

#     # Testing
#     res = mergeKLists([ListNode('')])
#     res_li = []

#     print(res)

#     'Notes: It worked'

'49. Group Anagrams'
# def x():

#     # Input

#     # Case 1
#     strs = ["eat","tea","tan","ate","nat","bat"]
#     #Exp. Out: [["bat"],["nat","tan"],["ate","eat","tea"]]

#     # Case 2
#     strs = [""]
#     #Exp. Out: [[""]]

#     # Case 3
#     strs = ["a"]
#     # Exp. Out: [["a"]]

#     # Custom Case
#     strs = ["ddddddddddg","dgggggggggg"]
#     # Expected: [["dgggggggggg"],["ddddddddddg"]]



#     'My Approach'

#     '''
#     Intuition:
#         1. Take the first element of the input and make a list with all element that contains the same characters
#         2. Erase the taken elements from the input.
#         3. Reiterate steps 1 & 2 until the input is exhausted

#     '''

#     def groupAnagrams(strs:list):
        
#         if len(strs) == 1:
#             return[strs]

#         # Auxiliary anagram checker
#         def is_anagram(ref:list, string:list):

#             if len(ref) != len(string):
#                 return False

#             for char in ref:
                
#                 if ref.count(char) != string.count(char):   
#                     return False

#             return True
        
#         # Creating Flag to manage repetitions
#         strs = [[word, False] for word in strs]


#         result = []

#         for word in strs:
                
#             if word[1] == False:

#                 anagrams = []
#                 anagrams.append(word[0])            
#                 word[1] = True

#                 for rest in strs:

#                     if rest[1] == False:

#                         if is_anagram(word[0], rest[0]):
#                             anagrams.append(rest[0])
#                             rest[1] = True
            
#                 result.append(anagrams)

#         return result
    
#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: 
#         It passed 72/126 cases, the case below broke the code: 
#             strs = ["ddddddddddg","dgggggggggg"] / Output: [["ddddddddddg","dgggggggggg"]], Expected: [["dgggggggggg"],["ddddddddddg"]]

#         After the fixture, it works but beat no one in efficiency
#     '''


#     'Another Approach'
#     def groupAnagrams(strs):
        
#         freq = {}

#         for word in strs:

#             newWord = ''.join(sorted(word))

#             if newWord not in freq:
#                 freq[newWord] = []
            
#             freq[newWord].append(word)

#         return list(freq.values())

#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: Absolutely more elegant solution
#     '''















