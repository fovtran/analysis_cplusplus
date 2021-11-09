import Control.Applicative
import Control.Monad
import Data.Either
import Math.NumberTheory.Powers.Squares

isInt :: RealFrac c => c -> Bool
isInt = (==) <$> id <*> fromInteger . round

intSqrt :: (Integral a) => a -> Int
--intSqrt = fromIntegral . floor . sqrt . fromIntegral
intSqrt = fromIntegral . integerSquareRoot'

factorize :: Int -> [Int]
factorize 1 = []
factorize n = first : factorize (quot n first)
  where first = (!! 0) $ [a | a <- [2..intSqrt n], rem n a == 0] ++ [n]

factorize2 :: Int -> [(Int,Int)]
factorize2 = foldl (\ls@((val,freq):xs) y -> if val == y then (val,freq+1):xs else (y,1):ls) [(0,0)] . factorize

numDivisors :: Int -> Int
numDivisors = foldl (\acc (_,y) -> acc * (y+1)) 1 <$> factorize2

nextTriangleNumber :: (Int,Int) -> (Int,Int)
nextTriangleNumber (n,acc) = (n+1,acc+n+1)

forward :: Int -> (Int, Int) -> Either (Int, Int) (Int, Int)
forward k val@(n,acc) = if numDivisors acc > k then Left val else Right (nextTriangleNumber val)

problem12 :: Int -> (Int, Int)
problem12 n = (!!0) . lefts . scanl (>>=) (forward n (1,1)) . repeat . forward $ n

main = do
  let (n,val) = problem12 1000
  print val
Using ghc -O3, this consistently runs in 0.55-0.58 seconds on my machine (1.73GHz Core i7).

A more efficient factorCount function for the C version:

int factorCount (int n)
{
  int count = 1;
  int candidate,tmpCount;
  while (n % 2 == 0) {
    count++;
    n /= 2;
  }
    for (candidate = 3; candidate < n && candidate * candidate < n; candidate += 2)
    if (n % candidate == 0) {
      tmpCount = 1;
      do {
        tmpCount++;
        n /= candidate;
      } while (n % candidate == 0);
       count*=tmpCount;
      }
  if (n > 1)
    count *= 2;
  return count;
  }
  
