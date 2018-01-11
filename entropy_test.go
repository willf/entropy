package entropy

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestNewNGramCounter(t *testing.T) {
	c := NewNGramCounter(2)
	if c.Size != 2 {
		t.Error("Didn't create a new ngram counter")
	}
}

func TestUpdate(t *testing.T) {
	v := NewNGramCounter(1)
	v.Update("1234")
	if v.Total != 4.0 {
		t.Error("update failed")
	}
}

func TestCountFound(t *testing.T) {
	v := NewNGramCounter(1)
	v.Update("1234")
	c := v.Count("1", 0.0)
	if c != 1.0 {
		t.Error("count found failed")
	}
}

func TestCountNotFound(t *testing.T) {
	v := NewNGramCounter(1)
	v.Update("1234")
	c := v.Count("Bob", 1.0)
	if c != 1.0 {
		t.Error("count found failed")
	}
}

func TestNewModel(t *testing.T) {
	m := New(3)
	if m.Size != 3 {
		t.Error("Didn't create a new model")
	}
	if len(m.Map) != 3 {
		t.Error("Didn't create a new model")
	}
}

func TestModelUpdate(t *testing.T) {
	m := New(3)
	m.Update("01234")
	if m.Map[3].Total != 3.0 {
		t.Error("update failed")
	}
}

func TestLogProb(t *testing.T) {
	m := New(3)
	m.Update("01234")
	lp := m.LogProb("012")
	if lp >= 0.0 {
		t.Errorf("Failed to get correct LogProb of 012: %v", lp)
	}
	lp = m.LogProb("z12")
	if lp >= 0.0 {
		t.Errorf("Failed to get correct LogProb of z12: %v", lp)
	}
	lp = m.LogProb("xxx")
	if lp >= 0.0 {
		t.Errorf("Failed to get correct LogProb of xxx: %v", lp)
	}
	lp = m.LogProb("")
	if lp != math.Inf(-1) {
		t.Errorf("Failed to get correct LogProb of '': %v", lp)
	}
	m = New(0)
	lp = m.LogProb("James Taylor")
	if lp != math.Inf(-1) {
		t.Errorf("Failed to get correct LogProb of '': %v", lp)
	}
}

func TestDump(t *testing.T) {
	m := New(3)
	m.Update("01234")
	var b bytes.Buffer
	m.Dump(&b)
	got := b.String()
	// fmt.Println(got)
	n := strings.Count(got, "\n")
	if n != 12 {
		t.Errorf("Expected 12 lines 5+4+3, but got %v", n)
	}
}

func TestPredict(t *testing.T) {
	m := New(3)
	m.Update("01234")
	p := m.Predict("01234")
	if p.Text != "01234" {
		t.Errorf("Forgot to copy text, expected 01234, got %v", p.Text)
	}
	if p.LogProbTotal >= 0 {
		t.Errorf("Forgot to calc LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage >= 0 {
		t.Errorf("Forgot to calc LogProbAverage,  got %v", p.LogProbAverage)
	}
	if p.NumberOfNGrams != 3 {
		t.Errorf("Forgot to set number of ngrams,  got %v", p.NumberOfNGrams)
	}
	p = m.Predict("")
	if p.LogProbTotal != 0 {
		t.Errorf("Forgot to calc LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage != 0 {
		t.Errorf("Forgot to calc LogProbAverage,  got %v", p.LogProbAverage)
	}
}

func TestRead(t *testing.T) {
	lines := `2	23	1
2	34	1
2	01	1
2	12	1
3	012	1
3	123	1
3	234	1
1	1	1
1	2	1
1	3	1
1	4	1
1	0	1
`
	reader := strings.NewReader(lines)
	model := Read(reader)
	if model.Size != 3 {
		t.Errorf("Didn't get the size right, expected 3, got %v", model.Size)
	}
	if len(model.Map) != 3 {
		t.Errorf("Didn't create three counters, got %v", len(model.Map))
	}
	// var b bytes.Buffer
	// model.Dump(&b)
	// got := b.String()
	// fmt.Println(got)
	p := model.Predict("01234")
	if p.Text != "01234" {
		t.Errorf("Forgot to copy text, expected 01234, got %v", p.Text)
	}
	if p.LogProbTotal >= 0 {
		t.Errorf("Forgot to calc LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage >= 0 {
		t.Errorf("Forg.ot to calc LogProbAverage,  got %v", p.LogProbAverage)
	}
	if p.NumberOfNGrams != 3 {
		t.Errorf("Forgot to set number of ngrams,  got %v", p.NumberOfNGrams)
	}
}

func TestReadInvalidLines(t *testing.T) {
	lines := `hey
boy
i
forgot
the
counts
`
	reader := strings.NewReader(lines)
	model := Read(reader)
	if model.Size != 0 {
		t.Errorf("Didn't get the size right, expected 0, got %v", model.Size)
	}
	p := model.Predict("01234")
	if p.Text != "01234" {
		t.Errorf("Forgot to copy text, expected 01234, got %v", p.Text)
	}
	if p.LogProbTotal != math.Log(0) {
		t.Errorf("Should be infinite LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage != math.Log(0) {
		t.Errorf("Should be infinite LogProbAverage,  got %v", p.LogProbAverage)
	}
	if p.NumberOfNGrams != 6 {
		t.Errorf("Forgot to set number of ngrams,  got %v", p.NumberOfNGrams)
	}
}

func TestTrain(t *testing.T) {
	lines := `1. Woody Guthrie
  2. The Weavers
  3. Bob Dylan
  4. Odetta
  5. Peter, Paul and Mary
  6. Pete Seeger
  7. Joni Mitchell
  8. Kingston Trio
  9. Joan Baez
10. Leadbelly
`
	reader := strings.NewReader(lines)
	model := New(2)
	model.Train(reader)
	if model.Size != 2 {
		t.Errorf("Didn't get the size right, expected 2, got %v", model.Size)
	}
	p := model.Predict("Paul")
	if p.Text != "Paul" {
		t.Errorf("Forgot to copy text, expected Paul, got %v", p.Text)
	}
	if p.LogProbTotal >= 0 {
		t.Errorf("Forgot to calc LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage >= 0 {
		t.Errorf("Forg.ot to calc LogProbAverage,  got %v", p.LogProbAverage)
	}
	if p.NumberOfNGrams != 3 {
		t.Errorf("Forgot to set number of ngrams,  got %v", p.NumberOfNGrams)
	}
}

func TestTrainWithMultiplier(t *testing.T) {
	lines := `,	27957346221
the	23688414489
.	19194317252
of	15342397280
and	11021132912
to	9494905988
in	7611765281
a	7083003595
"	4430963121
is	4139526351
`
	reader := strings.NewReader(lines)
	model := New(2)
	model.TrainWithMultiplier(reader)
	if model.Size != 2 {
		t.Errorf("Didn't get the size right, expected 2, got %v", model.Size)
	}
	p := model.Predict("Paul")
	if p.Text != "Paul" {
		t.Errorf("Forgot to copy text, expected Paul, got %v", p.Text)
	}
	if p.LogProbTotal >= 0 {
		t.Errorf("Forgot to calc LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage >= 0 {
		t.Errorf("Forg.ot to calc LogProbAverage,  got %v", p.LogProbAverage)
	}
	if p.NumberOfNGrams != 3 {
		t.Errorf("Forgot to set number of ngrams,  got %v", p.NumberOfNGrams)
	}
}

func TestTrainWithMultiplierBadLines(t *testing.T) {
	lines := `hey
girl
i
forgot
the
counts
`
	reader := strings.NewReader(lines)
	model := New(2)
	model.TrainWithMultiplier(reader)
	if model.Size != 2 {
		t.Errorf("Didn't get the size right, expected 2, got %v", model.Size)
	}
	p := model.Predict("Paul")
	if p.Text != "Paul" {
		t.Errorf("Forgot to copy text, expected Paul, got %v", p.Text)
	}
	if p.LogProbTotal < 0 {
		t.Errorf("Should be infinite LogProbTotal,  got %v", p.LogProbTotal)
	}
	if p.LogProbAverage < 0 {
		t.Errorf("Should be infinite LogProbAverage,  got %v", p.LogProbAverage)
	}
	if p.NumberOfNGrams != 3 {
		t.Errorf("Forgot to set number of ngrams,  got %v", p.NumberOfNGrams)
	}
}
