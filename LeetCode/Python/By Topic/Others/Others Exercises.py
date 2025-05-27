'''
CHALLENGES INDEX

7. Reverse Integer (Others)
8. String to Integer (atoi) (Others)
14. Longest Common Prefix (Others)
29. Divide Two Integers (Others)
38. Count and Say (Others)
66. Plus One (Others)
69. Sqrt(x) (Others)
149. Max Points on a Line (Others)
171. Excel Sheet Column Number (Others)
172. Factorial Trailing Zeroes (Others)
326. Power of Three (RC) (Others)


6. Zigzag Conversion (String)


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


'7. Reverse Integer'
# def x():

#     # Input
#     x = -15

#     # My Solution
#     raw_res = str(x)[::-1]

#     if '-' in raw_res:
#         res = int(raw_res[:-1])
#         res = int('-'+str(res))

#     else:
#         res = int(raw_res)


#     min_32bit = -2**31
#     max_32bit = 2**31-1

#     if res > max_32bit or res < min_32bit:
#         print(0)

#     else:
#         print(res)

'8. String to Integer (atoi)'
# def x():

#     # Input
#     s = "   -43.25"


#     # My approach

#     # Cleaning leading blank spaces
#     s = s.strip()

#     # In case of a sign present
#     sign = None

#     if s[0] == '-' or s[0] == '+':    
#         sign = s[0]
#         s = s[1:]


#     num = ''

#     # Reviewing each valid character
#     for char in s:    
#         if char not in '0123456789.':
#             break    
#         num += char


#     decimals = None

#     # In case of a decimals
#     if '.' in num:
#         decimals = num[num.find('.')+1:] #35
#         num = num[:num.find('.')]   #42
#         decimal_break = 5 * 10**(len(decimals)-1)

#         decimals = int(decimals)
        
#         #in case no number befor '.'
#         if not num:
#             num = 0
#         else:
#             num = int(num)
        
#         if decimals >= decimal_break:
#             num += 1

#     elif num:
#         num = int(num)



#     # In case is negative
#     if sign == '-':
#         num = int('-'+str(num))


#     max_32bit = 2**31-1
#     min_32bit = -2**31


#     #Outputting the result
#     if not num:
#         print(0)

#     else:
    
#         if num < min_32bit:
#             print(min_32bit)

#         elif num > max_32bit:
#             print(max_32bit)
        
#         else:      
#             print(num)

#     ''' 
#     Note:
#         It left cases unhandled. I also don't have the time to keep building the solution.
#     '''

#     # ChatGPT approach

#     def atoi(s: str) -> int:
#         s = s.strip()  # Remove leading and trailing whitespace
#         if not s:
#             return 0
        
#         sign = 1
#         i = 0
        
#         # Check for sign
#         if s[i] in ['+', '-']:
#             if s[i] == '-':
#                 sign = -1
#             i += 1
        
#         # Iterate through characters and build the number
#         num = 0
#         while i < len(s) and s[i].isdigit():
#             num = num * 10 + int(s[i])
#             i += 1
        
#         # Apply sign and handle overflow
#         num *= sign
#         num = max(-2**31, min(2**31 - 1, num))
        
#         return num

#     print(atoi(s))

'14. Longest Common Prefix'
# def x():

#     '''
#     Approach:

#         1. Order the array alphabetically 
#             & separate in different lists the words starting with each letter.

#         2. Order each array with the longest word upfront.

#         3. Build a dict with *Preffix as key and *Count as value.
#             *Count: will be how many words start with the first letter of the first word, the first two letters of the first word, 
#             and so on until exhauting the first (longest) word
#             *Preffix: the actual first, two first and so on substrings.

#         4. Merge all the resulting dict, order them descendingly, and return the first value if the count > 2, else: return an empty string.

#     '''

#     # Input

#     #   Custom input for testing
#     strs = ["flower", "flow", "flight", "dog", "racecar", "door", "fleet", "car", "racer"]

#     #   Real input
#     strs = ["a"]




#     # My approach

#     def longestCommonPrefix(strs):

#         strs = sorted(strs, reverse=True)

#         # Here will be stored each list
#         lists = {}

#         for str in strs:

#             first_letter = str[0]

#             if first_letter in lists:
#                 lists[first_letter].append(str)
            
#             else:
#                 lists[first_letter] = [str]


#         # Converting the dict into a list to facilitate the logic
#         groups = list(lists.values())

#         # Ordering each sublist by len
#         groups = list(map(lambda x: sorted(x, key=len, reverse=True), groups))

#         # Here will be the counting and the prefixes
#         results = dict()


#         for group in groups:

#             for i in range(1, len(group[0])):
                    
#                 prefix = ''
#                 count = 0

#                 for j in range(len(group)):

#                     if group[0][:i] in group[j]:
#                         count += 1
                    
#                 if count > 1:

#                     prefix = group[0][:i]
#                     results[prefix] = count


#         results = sorted(results.items(), key = lambda x: (x[1], x[0]), reverse=True)

#         # print(results)


#         if results:
#             return results[0][0]

#         else:
#             return ''
        

#     print(longestCommonPrefix(strs))

#     '''
#     Note:
#         My solution appears to be functional but is not working as expected with unexpected input.
#     '''

#     # ChatGPT's approach

#     def longestCommonPrefix(strs):

#         if not strs:
#             return ''

#         strs.sort()
#         first = strs[0]
#         last = strs[-1]

#         prefix = ''

#         for i in range(len(first)):

#             if i < len(last) and first[i] == last[i]:
                
#                 prefix += first[i]
            
#             else:
#                 break
        
#         return prefix


#     print(longestCommonPrefix(strs))

#     '''
#     Conclusion: The difference between my implementation and this is that the problem didn't state that the prefix must be present in all the strings, I assumed it wasn't going to be.
#     '''

'29. Divide Two Integers'
# def x():

#     # Input

#     # Case 1
#     dividend = 10
#     divisor = 3

#     # Case 2
#     dividend = 7
#     divisor = -3

#     # My approach

#     '''
#     Rationale:
#         1. Count how many times the divisor could be substracted from the dividend before reaching something smalle than the divisor
#         2. if only one between dividend and the divisor is less than 0, the result would return a negative number 
#     '''


#     def divide(dividend, divisor):
        
#         # case where 0 is divided by something
#         if dividend == 0:
#             return 0
        

#         # setting variables to operate
#         count = 0
#         div = abs(divisor)
#         dvnd = abs(dividend)


#         # case where the dividend is 1
#         if div == 1 and dvnd != 0:

#             if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#                 return -dvnd
            
#             else:
#                 return dvnd
        

#         # case where the absolute divisor is greater than the dividend
#         if div > dvnd:
#             return 0
        
#         # case where both are the same number
#         if div == dvnd:
                
#             if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#                 return -1
            
#             else:
#                 return 1
        
#         # case where is possible to divide iteratively
#         while dvnd >= div:

#             dvnd -= div
#             count += 1
        
#         # In case any is negative, the result will also be negative
#         if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#             return -count

#         # Otherwise, just return
#         return count


#     print(divide(dividend, divisor))

#     'Notes: My solution actually worked, theres nitpicking cases where it wont, but still '


#     # Another Approach

#     def divide(dividend, divisor):

#         if (dividend == -2147483648 and divisor == -1): return 2147483647
                
#         a, b, res = abs(dividend), abs(divisor), 0

#         for x in range(32)[::-1]:

#             if (a >> x) - b >= 0:
#                 res += 1 << x
#                 a -= b << x
        
#         return res if (dividend > 0) == (divisor > 0) else -res

#     'Notes: This challenge is solved with bitwise operations '

'38. Count and Say'
# def x():

#     # Input

#     # Case 1
#     n = 1   # Exp. Out: "1" (Base Case)

#     # Case 2
#     n = 4   # Exp. Out: "1211" (Base Case)


#     'My Approach - Iterative suboptimal solution' 
#     def countAndSay(n):

#         if n == 1:
#             return '1'
            
#         res = '1'

#         for _ in range(1, n):
        
#             pairs = []
#             count = 0
#             char = res[0]

#             for i in range(len(res)+1):

#                 if i == len(res):
#                     pairs.append(str(count)+char)

#                 elif res[i] == char:
#                     count += 1

#                 else:       
#                     pairs.append(str(count)+char)
#                     char = res[i]
#                     count = 1

#             res = ''.join(pairs)
        
#         return res

#     # Testing
#     print(countAndSay(6))

#     'Notes: It works'

#     'Recursive Approach'
#     def countAndSay(n):
#         if n == 1:
#             return '1'
#         return aux_countAndSay(countAndSay(n - 1))

#     def aux_countAndSay(s):
    
#         if not s:
#             return ''
        
#         result = []
#         count = 1

#         for i in range(1, len(s)):

#             if s[i] == s[i - 1]:
#                 count += 1

#             else:
#                 result.append(str(count) + s[i - 1])
#                 count = 1

#         result.append(str(count) + s[-1])

#         return ''.join(result)

#     # Testing
#     print(countAndSay(6))

#     'Done'

'66. Plus One'
# def x():

#     # Input
#     # Case 1
#     digits = [1,2,3]
#     # Output: [1,2,4]

#     # Case 2
#     digits = [4,3,2,1]
#     # Output: [4,3,2,2]

#     # Case 3
#     digits = [9]
#     # Output: [1,0]

#     # Custom Case
#     digits = [9,9,9]
#     # Output: [1,0,0,0]

#     '''
#     My Approach
#         Intuition:
#             - The case is simple, the catch is to handle the case "[9,9,9]"
#     '''

#     def plusOne(digits: list[int]) -> list[int]:

#         idx = -1

#         while abs(idx) <= len(digits):
            
#             if abs(idx) == len(digits) and digits[idx] == 9:

#                 digits[idx] = 1
#                 digits.append(0)
#                 break

#             if digits[idx] != 9:

#                 digits[idx] += 1
#                 break

#             digits[idx] = 0
#             idx -= 1

#         return digits

#     # Testing
#     print(plusOne(digits=digits))

#     '''
#     Notes: 

#         While this code works, there was an even cleverer approach - To convert the digits into a int, add 1 and return as a list of ints
#         this way, is avoided the handling of cases
#     '''


#     'A different Approach'
#     def plusOne(digits: list[int]) -> list[int]:

#         number = int(''.join([str(x) for x in digits]))
#         number += 1
        
#         return [int(x) for x in str(number)]

#     # Testing
#     print(plusOne(digits=digits))

'69. Sqrt(x)'
# def x():

#     # Input
#     # Case 1
#     x = 4
#     # Output: 2

#     # Case 2
#     x = 8
#     # Output: 2

#     # Custom Case
#     x = 399
#     # Output: ..

#     'My Approach'

#     limit = 46341

#     # Auxiliary Eratosthenes' sieve function
#     def primes(cap):  

#         primes = []
#         not_primes = []

#         for i in range(2, cap+1):

#             if i not in not_primes:
#                 primes.append(i)
#                 not_primes.extend([x for x in range(i*i, cap+1, i)])

#         return primes

#     def mySqrt(x:int) -> int:

#         #Setting a limit for calculating primes
#         limit = x//2

#         prime_nums = primes(limit)

#         squares = list(map(lambda x: x*x, prime_nums))


#         #proning in the squares the correct range to make a range to evaluate
#         root_range = []
#         for idx, v in enumerate(squares):

#             if x <= v:
#                 root_range = [prime_nums[idx-1], prime_nums[idx]]
#                 break

#         #Calculating manually the square of each root in range to select the floor-root for the value
#         for root in range(root_range[0], root_range[1]+1):
            
#             if root*root >= x:
#                 return result
            
#             result = root

#     # Testing
#     print(mySqrt(x))

#     'Notes: This approach was too complicated and actually not as efficient. Apparently with the notion of binary search is easier to solve'


#     'Binary Search Approach'
#     def mySqrt(x):

#         left = 0
#         right = x

#         while left <= right:

#             mid = (left + right)//2

#             if mid*mid < x:
#                 left = mid + 1

#             elif mid*mid > x: 
#                 right = mid -1

#             else:
#                 return mid
        
#         return right

#     # Testing
#     print(mySqrt(x))

'''149. Max Points on a Line'''
# def x():

#     '''
#     Revision

#         The problem could be pretty hard if no math knowledge is acquired beforehand.
#         By definition, if several points share the same 'slope' with one single point,
#         it'd mean that they are all included in the same line.

#         So the problem reduces to (brut force) check for each point if the rest share the same
#         slope and the biggest group with common slope will be the answer
#     '''

#     def maxPoints(points:list[list[int]]):

#         # if there is no more than a pair of point in the plane, well, that's the answer
#         if len(points) < 3:
#             return len(points)
        
#         # Initializing with the lowest possible answer
#         result = 2

#         # Since we are counting on pairs, we're iterating up to the second last point of the group
#         for i, point1 in enumerate(points[:-1]):

#             slopes = {} # The keys will be the slopes and the values the count of points with the same slope

#             for point2 in points[i+1:]:
                
#                 slope = None
#                 x_comp = point2[0] - point1[0]

#                 if x_comp:  # The bool of 0 is False
                    
#                     # Calculate the slope
#                     slope = (point2[1] - point1[1]) / x_comp

#                 # If that slope already exist, add one point to the count
#                 if slope in slopes:

#                     slopes[slope] += 1
#                     new = slopes[slope]

#                     result = max(result, new)
                
#                 # else, create a new dict entry
#                 else:
#                     slopes[slope] = 2

#         return result

#     'Done'

'''171. Excel Sheet Column Number'''
# def x():

#     # Input
#     # Case 1
#     columnTitle = 'A'
#     # Output: 1

#     # Case 2
#     columnTitle = 'AB'
#     # Output: 28

#     # Case 3
#     columnTitle = 'ZY'
#     # Output: 701

#     # Custom Case
#     columnTitle = 'ASY'
#     # Output: 1195

#     'My Approach'
#     def titleToNumber(columnTitle: str) -> int:
    
#         dic = {v:k for k,v in list(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 1))}
#         res = 0

#         for idx, letter in enumerate(reversed(columnTitle)):
#             res += dic[letter]*pow(26, idx)
        
#         return res

#     # Testing
#     print(titleToNumber(columnTitle=columnTitle))

#     'Done'

'''172. Factorial Trailing Zeroes'''
# def x():

#     # Input
#     # Case 1
#     n = 3
#     # Output: 0 (3! = 6, no trailing zero).

#     # Case 2
#     n = 5
#     # Output: 1 (5! = 120).

#     # Case 3
#     n = 0
#     # Output: 0 (0! = 1).

#     # Custom case
#     n = 1574
#     # Output: 390 


#     'My Approach'
#     def trailingZeroes(n: int) -> int:

#         res = 1

#         for i in range(2, n+1):
#             res *= i
        
#         zeros = 0

#         while True:

#             if  res % 10 != 0:
#                 break
            
#             zeros += 1
#             res //= 10    
            
#         return zeros

#     # Testing
#     print(trailingZeroes(n=1574))

#     'Note: While my approach works and passed, is not as efficient, is O(n)'


#     '''
#     Optimized approach

#         Taking advantage of the fact that every factor of 5 contributes to trailing zeros
#         the problem simplifies greatly since no factorials are needed to be calculated
#     '''

#     def trailingZeroes(n: int) -> int:

#         zeros = 0

#         while n >= 5:

#             n //= 5
#             zeros += n          
            
#         return zeros

#     # Testing
#     print(trailingZeroes(n=1574))

#     'Done'

'''326. Power of Three'''
# def x():

#     # Input
#     # Case 1
#     n = 45
#     # Output: True

#     # Custom Case
#     n = -1
#     # Output: True


#     'Iterative approach'
#     def is_power_of_three(n:int) -> bool:

#         powers = [3**x for x in range(21)]

#         return n in powers


#     'Recursive apporach'
#     def is_power_of_three(n:int) -> bool:

#         # Base case: if n is 1, it's a power of three
#         if n == 1:
#             return True

#         # If n is less than 1, it can't be a power of three
#         if n < 1:
#             return False

#         # Recursive case: check if n is divisible by 3 and then recurse with n divided by 3
#         if n % 3 == 0:
#             return is_power_of_three(n // 3)

#         # If n is not divisible by 3, it's not a power of three
#         return False

#     'Done'




"""6. Zigzag Conversion"""
# def x():
    
#     # Input
#     # Case 1
#     s = "PAYPALISHIRING"
#     numRows = 3
#     # Output: "PAHNAPLSIIGYIR"

#     # Case 2
#     s = "PAYPALISHIRING"
#     numRows = 4
#     # Output: "PINALSIGYAHRPI"

#     # Case 3
#     s = "A"
#     numRows = 1
#     # Output: "A"

#     # Case 4
#     s = ""
#     numRows = 1
#     # Output: ""

#     # Case 5
#     s = "AB"
#     numRows = 1
#     # Output: "AB"

#     # Case 6
#     s = "AB"
#     numRows = 2
#     # Output: "AB"

#     # Case 7
#     s = "AB"
#     numRows = 3
#     # Output: "AB"

#     # Case 8
#     s = "ABCD"
#     numRows = 4
#     # Output: "ABCD"

#     '''
#     EXPLANATION

#         Step-by-step:

#             - Initialize rows = [""] * numRows

#             - Track curr_row (current row) and going_down (direction).

#             - For each character:

#                 + Append it to the current row: rows[curr_row] += char

#                 + If at the top or bottom row, flip direction

#                 + Move curr_row accordingly: +1 if down, -1 if up

#             - At the end, return ''.join(rows)
#     '''

#     def convert(s: str, numRows: int) -> str:

#         # Handle Corner Case: Not enough columns to break the word
#         if numRows == 1 or numRows >= len(s):
#             return s

#         # Initializing the working variables
#         rows = [''] * numRows
#         curr_row = 0
#         going_down = False

#         # Traverse the string
#         for char in s:

#             rows[curr_row] += char

#             # Change direction if we hit the top or bottom row
#             if curr_row == 0 or curr_row == numRows - 1:
#                 going_down = not going_down

#             # Move up or down
#             curr_row += 1 if going_down else -1

#         # Join all the rows and return the resulting string
#         return ''.join(rows)

#     # Testing
#     print(convert(s=s, numRows=numRows))

#     '''Note: Done'''



























