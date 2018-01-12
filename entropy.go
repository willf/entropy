package entropy

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strings"
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
	counter.UpdateWithMultiplier(line, 1)
}

// UpdateWithMultiplier updates the counter for a string, using
// a multiplier
func (counter *NGramCounter) UpdateWithMultiplier(line string, multiplier uint64) {
	for _, key := range sliding([]rune(line), counter.Size) {
		counter.Counts[string(key)] += multiplier
		counter.Total += multiplier
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
	Size    int
	Counter *NGramCounter
}

// New creates a Model with maximum ngram size of `MaxNGramSize`
func New(MaxNGramSize int) (model *Model) {
	model = new(Model)
	model.Size = MaxNGramSize
	model.Counter = NewNGramCounter(MaxNGramSize)
	return
}

// Update for Models send string to each counter
func (model *Model) Update(line string) {
	model.UpdateWithMultiplier(line, 1)
}

// UpdateWithMultiplier for Models send string to each counter with multiplier
func (model *Model) UpdateWithMultiplier(line string, multiplier uint64) {
	model.Counter.UpdateWithMultiplier(line, multiplier)
}

// LogProb returns the best matching log probability for a key given
// a set of models
func (model *Model) LogProb(key string) (logProb float64) {
	if model.Size == 0 || len(key) == 0 {
		logProb = math.Inf(-1)
		return
	}
	counter := model.Counter
	count := counter.Count(key, 0)
	if count == 0 {
		logProb = math.Log2(0.5) - math.Log2(float64(counter.Total))
	} else {
		logProb = math.Log2(float64(count)) - math.Log2(float64(counter.Total))
	}

	return
}

// Dump sends a set of ngram models to a writer
func (model *Model) Dump(f io.Writer) {
	counter := model.Counter
	sz := model.Size
	for key, value := range counter.Counts {
		outs := fmt.Sprintf("%v\t%s\t%d\n", sz, key, value)
		_, err := f.Write([]byte(outs))
		if err != nil {
			panic(err)
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
	var counter *NGramCounter
	var ngramSize int
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
		if ngramSize == 0 {
			ngramSize = size
			counter = NewNGramCounter(ngramSize)
		}
		// fmt.Printf("size: %v, ngram: '%v', count: %v\n", size, ngram, count)
		counter.Total += count
		counter.Counts[ngram] = count
	}
	model.Size = ngramSize
	model.Counter = counter
	return
}

// Train trains a set of ngram models from a file. Models must be
// initialized. returns the number of example lines used
func (model *Model) Train(f io.Reader) (exampleCount int) {
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		text := sc.Text()
		exampleCount++
		model.Update(strings.TrimSpace(text))
	}
	return
}

// TrainWithMultiplier trains a set of ngram models from a file. Models must be
// initialized. returns the number of example lines used.
// format is token <tab> count
func (model *Model) TrainWithMultiplier(f io.Reader) (exampleCount int) {
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		text := sc.Text()
		exampleCount++
		var ngram string
		var count uint64
		_, err := fmt.Sscanf(text, "%s\t%d", &ngram, &count)
		if err != nil {
			fmt.Printf("Invalid line at %v: %v\n", exampleCount, text)
		}
		model.UpdateWithMultiplier(text, count)
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
