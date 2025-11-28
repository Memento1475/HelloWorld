#include <stdio.h>
#define max_size 1000

int main() {
    int n, m;
    int count = 0;
    scanf_s("%d %d", &n, &m);

    if (n < m || m < 2 || n > max_size) {
        return 1;
    }

    int remaining = n;

    int a[max_size] = {0};
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }

    int i = 0;
     while (remaining > 1) {
         if (a[i] == 0) {
             count++;

             if (count == m) {
                 a[i] = 1;
                 count = 0;
                 remaining--;
             }
         }
         i = (i + 1) % n;
     }

     for (int j = 0; j < n; j++) {
         if (a[j] == 0) {
             printf("%d", j + 1);
         }    
     }

    return 0;
}
