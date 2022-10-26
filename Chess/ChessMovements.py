''' Test script to compute all edges of chess board problem '''

__author__ = "Hayato Kato"

def main():
	print('starting')
	V_knight = []
	for i in range(8):
		for j in range(8):
			V_knight.append((i,j))

	print(len(V_knight))

	E_knight = []
	for start_v in V_knight:
		i,j = start_v
		end_v = []
		end_v.append((i+1,j+2))
		end_v.append((i-1,j+2))
		end_v.append((i+1,j-2))
		end_v.append((i-1,j-2))
		end_v.append((i+2,j+1))
		end_v.append((i-2,j+1))
		end_v.append((i+2,j-1))
		end_v.append((i-2,j-1))
		for end in end_v:
			print(end)
			if end in V_knight:
				E_knight.append((start_v, end))

	print(len(E_knight))

if __name__ == '__main__':
	main()