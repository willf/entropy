package entropy

import (
	"bufio"
	"fmt"
	"io"
	"math"
)

// NGramCounter contains counts and totals for Ngrams of a
// particular size
type NGramCounter struct {
	Size   int
	Counts map[string]uint64
	Total  uint64
}

// NewNGramCounter returns a new ngram counter
func NewNGramCounter(maxNGramSize int) (counter *NGramCounter) {
	counter = new(NGramCounter)
	counter.Size = maxNGramSize
	counter.Counts = make(map[string]uint64)
	return
}

// Prediction is the log probability of a string and other data
type Prediction struct {
	LogProbAverage float64
	LogProbTotal   float64
	NumberOfNGrams int
	Text           string
}

// Update updates the counter for a newly seen string
func (counter *NGramCounter) Update(line string) {
	for _, key := range sliding([]rune(line), counter.Size) {
		counter.Counts[string(key)] += 1.0
		counter.Total += 1.0
	}
}

// Count returns the number of ngrams in a particular counter.
// returns default if not found
func (counter *NGramCounter) Count(key string, ifNotFound uint64) (count uint64) {
	count, ok := counter.Counts[key]
	if !ok {
		count = ifNotFound
	}
	return
}

// Model contains a max size and a map from its to character models
type Model struct {
	Size int
	Map  map[int]*NGramCounter
}

// New creates a Model with maximum ngram size of `MaxNGramSize`
func New(MaxNGramSize int) (model *Model) {
	model = new(Model)
	model.Size = MaxNGramSize
	model.Map = make(map[int]*NGramCounter)
	for key := 1; key <= MaxNGramSize; key++ {
		model.Map[key] = NewNGramCounter(key)
	}
	return
}

// Update for Models send string to each counter
func (model *Model) Update(line string) {
	for _, counter := range model.Map {
		counter.Update(line)
	}
}

// LogProb returns the best matching log probability for a key given
// a set of models
func (model *Model) LogProb(key string) (logProb float64) {
	if model.Size == 0 || len(key) == 0 {
		logProb = math.Inf(-1)
		return
	}
	var lastTotal uint64
	// find the table that is the same size as the key
	// looking at increasing shorter suffixes
	// e.g. abc, bc, c
	runes := []rune(key)
	for i := 0; i < len(runes); i++ {
		key := string(runes[i:])
		counter, foundCounter := model.Map[len(key)]
		if foundCounter {
			count := counter.Count(key, 0.0)
			if count > 0.0 {
				logProb = math.Log2(float64(count)) - math.Log2(float64(counter.Total))
				return
			}
			lastTotal = counter.Total
		}
	}
	// found it nowhere ... use last Total, and '1 count'
	logProb = math.Log2(1.0) - math.Log2(float64(lastTotal))
	return
}

// Dump sends a set of ngram models to a writer
func (model *Model) Dump(f io.Writer) {
	for sz, counter := range model.Map {
		for key, value := range counter.Counts {
			outs := fmt.Sprintf("%v\t%s\t%d\n", sz, key, value)
			_, err := f.Write([]byte(outs))
			if err != nil {
				panic(err)
			}
		}
	}
}

// Predict returns a prediction for a string
func (model *Model) Predict(text string) (prediction *Prediction) {
	prediction = new(Prediction)
	prediction.Text = text
	keys := sliding([]rune(text), model.Size)
	nKeys := len(keys)
	prediction.NumberOfNGrams = nKeys
	if len(keys) == 0 {
		return
	}
	var logProbTotal float64
	for _, key := range keys {
		lp := model.LogProb(key)
		logProbTotal += lp
	}
	logProbAverage := logProbTotal / float64(nKeys)
	prediction.LogProbAverage = logProbAverage
	prediction.LogProbTotal = logProbTotal
	return
}

// Read reads a model in
func Read(f io.Reader) (model *Model) {
	model = new(Model)
	scanner := bufio.NewScanner(f)
	counterMap := make(map[int]*NGramCounter)
	var maxSize int
	var linenum uint
	for scanner.Scan() {
		linenum++
		var size int
		var ngram string
		var count uint64
		text := scanner.Text()
		_, err := fmt.Sscanf(text, "%v\t%s\t%d", &size, &ngram, &count)
		if err != nil {
			fmt.Printf("Invalid line at %v: %v\n", linenum, text)
		}

		// fmt.Printf("size: %v, ngram: '%v', count: %v\n", size, ngram, count)
		counter, ok := counterMap[size]
		if !ok {
			counter = NewNGramCounter(size)
			counterMap[size] = counter
		}
		counter.Total += count
		counter.Counts[ngram] = count
		if size > maxSize {
			maxSize = size
		}
	}
	model.Size = maxSize
	model.Map = counterMap
	return
}

// Train trains a set of ngram models from a file. Models must be
// initialized. returns the number of example lines used
func (model *Model) Train(f io.Reader) (exampleCount int) {
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		text := sc.Text()
		exampleCount++
		model.Update(text)
	}
	return
}

// sliding window function
func sliding(s []rune, length int) (windows []string) {
	for i := 0; i+length <= len(s); i++ {
		windows = append(windows, string(s[i:i+length]))
	}
	return
}

//
// // var maxNGramSize = flag.Int("MaxNGramSize", 3, "Ngram size")
// // var train = flag.Bool("train", false, "Set if you want to train")
// // var predict = flag.Bool("predict", false, "Set if you want to predict")
// // var infile = flag.String("in", "", "Set for reading")
// // var outfile = flag.String("out", "", "Set for output")
// // var modelfile = flag.String("model", "", "Set for model file (dump or read)")
// //
// // func main() {
// // 	flag.Parse()
// // 	ok := (*train || *predict)
// // 	if !ok {
// // 		fmt.Println("Either train or predict must be specified")
// // 		return
// // 	}
// //
// // 	models := NewModels(*maxNGramSize)
// //
// // 	inf := os.Stdin
// // 	if *infile != "" {
// // 		f2, err := os.Open(*infile)
// // 		if err != nil {
// // 			fmt.Printf("error reading %s\n", *infile)
// // 			return
// // 		}
// // 		inf = f2
// // 		defer inf.Close()
// // 	}
// //
// // 	if *train {
// //
// // 		modf := os.Stdout
// // 		if *modelfile != "" {
// // 			f2, err := os.Create(*modelfile)
// // 			if err != nil {
// // 				fmt.Printf("error creating %s\n", *modelfile)
// // 				return
// // 			}
// // 			modf = f2
// // 			defer modf.Close()
// // 		}
// // 		models.Train(inf)
// // 		models.Dump(modf)
// // 		return
// // 	}
// //
// // 	if *predict {
// // 		modf := os.Stdin
// // 		outf := os.Stdout
// // 		if *modelfile != "" {
// // 			f2, err := os.Open(*modelfile)
// // 			if err != nil {
// // 				fmt.Printf("error reading %s\n", *modelfile)
// // 				return
// // 			}
// // 			modf = f2
// // 		}
// // 		if *outfile != "" {
// // 			f2, err := os.Create(*outfile)
// // 			if err != nil {
// // 				fmt.Printf("error creating %s\n", *outfile)
// // 				return
// // 			}
// // 			outf = f2
// // 			defer outf.Close()
// // 		}
// //
// // 		models.Read(modf)
// // 		models.Predict(inf, outf)
// // 		return
// // 	}
// //
// // 	// Just setting this here so I can use this this source as an example..
// // 	magicPassword := "PXKXoyThngGrjCgBLuf2ivrpFFNKA9UgBHrxpLaW"
// // 	if magicPassword == "" {
// // 		fmt.Println("this would be surprising to see")
// // 	}
// // 	standardPassword := "escalate-latrine-footed-ping"
// // 	if standardPassword == "" {
// // 		fmt.Println("this would be surprising to see")
// // 	}
// // 	stupidPassword := "letmein"
// // 	if stupidPassword == "" {
// // 		fmt.Println("this would be surprising to see")
// // 	}
// //
// // }
