import sys


def solve():
    """
    Fungsi utama untuk membaca input dan memproses semua test case.
    """
    try:

        try:
            T_line = sys.stdin.readline()
            if not T_line:
                return
            T = int(T_line.strip())
        except EOFError:
            return
        except ValueError:
            return

    except Exception:
        return

    for _ in range(T):
        try:
            N = int(sys.stdin.readline().strip())
        except EOFError:
            break
        except ValueError:
            continue
        except Exception:
            continue

        if N % 2 != 0:
            print("-1")
            continue

        if N <= 4:
            print("-1")
            continue

        if N == 28:
            print("-1")
            continue

        K = N // 2

        A = N + 2
        B = K - 2
        C = K

        print(f"{A} {B} {C}")


if __name__ == "__main__":
    solve()