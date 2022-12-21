from math import floor
import re

map1 = {
  0: ["zero", "zerowy"],
  1: ["jeden", "pierwszy"],
  2: ["dwa", "drugi"],
  3: ["trzy", "trzeci"],
  4: ["cztery", "czwarty"],
  5: ["pięć", "piąty"],
  6: ["sześć", "szósty"],
  7: ["siedem", "siódmy"],
  8: ["osiem", "ósmy"],
  9: ["dziewięć", "dziewiąty"],
}

map2 = {
  10: ["dziesięć", "dziesiąty"],
  11: ["jedenaście", "jedenasty"],
  12: ["dwanaście", "dwunasty"],
  13: ["trzynaście", "trzynasty"],
  14: ["czternaście", "czternasty"],
  15: ["piętnaście", "piętnasty"],
  16: ["szesnaście", "szesnasty"],
  17: ["siedemnaście", "siedemnasty"],
  18: ["osiemnaście", "osiemnasty"],
  19: ["dziewiętnaście", "dziewiętnasty"],
}

map3 = {
  2: ["dwadzieścia", "dwudziesty"],
  3: ["trzydzieści", "trzydziesty"],
  4: ["czterdzieści", "czterdziesty"],
  5: ["pięćdziesiąt", "pięćdziesiąty"],
  6: ["sześćdziesiąt", "sześćdziesiąty"],
  7: ["siedemdziesiąt", "siedemdziesiąty"],
  8: ["osiemdziesiąt", "osiemdziesiąty"],
  9: ["dziewięćdziesiąt", "dziewięćdziesiąty"],
}

map4 = {
  1: ["sto", "setny"],
  2: ["dwieście", "dwusetny"],
  3: ["trzysta", "trzysetny"],
  4: ["czterysta", "czterysetny"],
  5: ["pięćset", "pięćsetny"],
  6: ["sześćset", "sześćsetny"],
  7: ["siedemset", "siedemsetny"],
  8: ["osiemset", "osiemsetny"],
  9: ["dziewięćset", "dziewięćsetny"],
}

map5 = {
  1: ["tysiąc", "tysięczny"],
  2: ["dwa tysiące", "dwutysięczny"],
}

def _convertToWords(input: str):  
  def _convert(num: int):
    if (num < 10): return map1[num]
    if (num < 20): return map2[num]
    if (num < 100): 
      tens = floor(num / 10)
      units = num % 10
      part1 = [map3[tens][0], map1[units][0] if units > 0 else ""]
      part2 = [map3[tens][1], map1[units][1] if units > 0 else ""]

      return [" ".join(filter(lambda x: len(x) > 0, part1)), " ".join(filter(lambda x: len(x) > 0, part2))]


    if (num < 1000):
      hundreds = floor(num / 100)
      tensAndUnits = num - hundreds * 100
      if tensAndUnits == 0: return map4[hundreds]
      return list(map(lambda x: f'{map4[hundreds][0]} {x}', _convertToWords(tensAndUnits)))
    

    if (num < 3000):
        thousands = floor(num / 1000)
        remaining = num - thousands * 1000
        if remaining == 0: return map5[thousands]
        return list(map(lambda x: f'{map5[thousands][0]} {x}', _convertToWords(remaining)))

    return ["<UNSUPPORTED>", "<UNSUPPORTED>"]

  toProcess = int(input)
  prefix = "minus " if toProcess < 0 else ""
  num = abs(toProcess)
  return list(map(lambda x: prefix + x, _convert(num)))


_number_re = re.compile(r"[0-9]+'s|[0-9]+s|[0-9]+\.?")
_roman_re = re.compile(r'\b(?=[MDCLXVI]+\b)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{2,3})\b')  # avoid I

def _expand_roman(m):
    # from https://stackoverflow.com/questions/19308177/converting-roman-numerals-to-integers-in-python
  roman_numerals = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
  result = 0
  num = m.group(0)
  for i, c in enumerate(num):
      if (i+1) == len(num) or roman_numerals[c] >= roman_numerals[num[i+1]]:
          result += roman_numerals[c]
      else:
          result -= roman_numerals[c]
  return str(result) + '.'

def _expand_number(m):
  _, number, suffix = re.split(r"(\d+(?:'?\d+)?)", m.group(0))
  return _convertToWords(number)[0 if suffix == '' else 1]

def normalize_numbers(text: str):
  text = re.sub(_roman_re, _expand_roman, text)
  text = re.sub(_number_re, _expand_number, text)
  return text
