<?php

class NN
{
  public $epochs = 10;

  /* 
   * starting weights (usually random) 
   * both weights and bias neet to shape as same the inputs
   */

  public $hidden1_weights = [[0.00905953, -0.00076626], [-0.02411565,  0.00856424]];
  public $hidden1_bias = [0.0, 0.0];

  public $hidden2_weights = [[0.01191972,  0.0147446], [0.00220843, -0.0027133]];
  public $hidden2_bias = [0.0, 0.0];

  public $learningRate = 1.0;

  /*
   * Feedforward the input and return sotmax output
   */
  public function predict($input)
  {
    /*
     * dot product hidden1 layer
     * dot product with inputs and hidden1 layer weights and bias
     */
    $dot_hidden_1 = $this->hidden1_bias;

    for ($i = 0; $i < count($dot_hidden_1); $i++) {
      $dot_hidden_1[$i] += $input[0] * $this->hidden1_weights[0][$i];
      $dot_hidden_1[$i] += $input[1] * $this->hidden1_weights[1][$i];
    }

    /* 
     * ReLU activation
     */
    $dot_hidden_1_relu = [];

    for ($i = 0; $i < count($dot_hidden_1); $i++) {
      $dot_hidden_1_relu[$i] = ReLU($dot_hidden_1[$i]);
    }

    /*
     * dot product hidden2 layer
     * dot product with layer1 outputs (with the activation function) and layer2 weights and bias
     */
    $dot_hidden_2 = $this->hidden2_bias;

    for ($i = 0; $i < count($dot_hidden_2); $i++) {
      $dot_hidden_2[$i] += $dot_hidden_1_relu[0] * $this->hidden2_weights[0][$i];
      $dot_hidden_2[$i] += $dot_hidden_1_relu[1] * $this->hidden2_weights[1][$i];
    }

    /* softmax activation 
     * probability distribution between the labels
     */
    return softmax($dot_hidden_2);
  }

  /*
   * Feedforward and Backpropagation with naive optimizing
   */
  public function fit($inputs, $labels)
  {
    $e = 0;
    while ($e < $this->epochs) {

      /* 
       * dot product hidden1 layer
       * dot product with inputs and hidden1 layer weights and bias
       */
      $dot_hidden_1 = [];

      foreach ($inputs as $input) {
        $d_row = $this->hidden1_bias;

        for ($i = 0; $i < count($d_row); $i++) {
          $d_row[$i] += $input[0] * $this->hidden1_weights[0][$i];
          $d_row[$i] += $input[1] * $this->hidden1_weights[1][$i];
        }

        $dot_hidden_1[] = $d_row;
      }

      /* 
       * ReLU activation
       */
      $dot_hidden_1_relu = array_map(function ($x) {
        $y = [];
        foreach ($x as $z) {
          $y[] = ReLU($z);
        }

        return $y;
      }, $dot_hidden_1);

      /*
       * dot product hidden2 layer
       * dot product with layer1 outputs (with the activation function) and layer2 weights and bias
       */
      $dot_hidden_2 = [];
      foreach ($dot_hidden_1_relu as $input) {
        $d_row = $this->hidden2_bias;

        for ($i = 0; $i < count($d_row); $i++) {
          $d_row[$i] += $input[0] * $this->hidden2_weights[0][$i];
          $d_row[$i] += $input[1] * $this->hidden2_weights[1][$i];
        }

        $dot_hidden_2[] = $d_row;
      }

      /* 
       * softmax activation
       */
      $softmax_output = [];
      foreach ($dot_hidden_2 as $input) {
        $softmax_output[] = softmax($input);
      }

      /* 
       * calculating loss
       */
      $loss = categ_cross_entropy($softmax_output, $labels);

      /* 
       * calculating accuracy
       */
      $predictions = [];
      foreach ($softmax_output as $out) {
        $predictions[] = array_keys($out, max($out))[0];
      }

      $y_true = [];
      foreach ($labels as $label) {
        $y_true[] = array_keys($label, max($label))[0];
      }

      $accuracy = [];
      for ($i = 0; $i < count($predictions); $i++) {
        $accuracy[] = $predictions[$i] == $y_true[$i];
      }
      $accuracy = array_sum($accuracy) / count($accuracy);

      /* backpropagation
       * loss gradient backpropagation
       * starting with loss derivative
       */
      $d_loss = categ_cross_entropy_d($softmax_output, $labels);

      /* 
       * hidden2 derivatives
       */
      $dweights_l2 = [[0, 0], [0, 0]];
      $dinputs_l2 = [];
      $dbias_l2 = [0, 0];

      /* 
       * hidden2 derivatives in respect to each weight (inputs.T * derivatives)
       */
      $dweights_l2[0][0] = dot_product(($dot_hidden_1_relu)[0], $d_loss[0]);
      $dweights_l2[0][1] = dot_product(transpose($dot_hidden_1_relu)[1], $d_loss[0]);

      $dweights_l2[1][0] = dot_product(($dot_hidden_1_relu)[1], $d_loss[1]);
      $dweights_l2[1][1] = dot_product(transpose($dot_hidden_1_relu)[1], $d_loss[1]);

      /* 
       * hidden2 derivatives in respect to each input (derivatives * weights.T)
       */
      for ($i = 0; $i < count($d_loss); $i++) {
        $dinputs_l2[$i][0] = dot_product($this->hidden2_weights[0], $d_loss[$i]);
        $dinputs_l2[$i][1] = dot_product($this->hidden2_weights[1], $d_loss[$i]);
      }

      /* 
       * hidden2 derivatives in respect to bias
       */
      for ($z = 0; $z < count($d_loss); $z++) {
        $dbias_l2[0] += $d_loss[$z][0];
        $dbias_l2[1] += $d_loss[$z][1];
      }

      /* 
       * ReLU derivative
       */
      $d_relu = [];
      for ($i = 0; $i < count($dot_hidden_1); $i++) {
        $w0 = $dot_hidden_1[$i][0] <= 0 ? 0 : $dinputs_l2[$i][0];
        $w1 = $dot_hidden_1[$i][1] <= 0 ? 0 : $dinputs_l2[$i][1];

        $d_relu[] = [$w0, $w1];
      }

      /*  
       * hidden2 derivatives
       */
      $dweights_l1 = [[0, 0], [0, 0]];
      $dinputs_l1 = [];
      $dbias_l1 = [0, 0];

      /* 
       * hidden1 derivatives in respect to each weight (inputs.T * derivatives)
       */
      $dweights_l1[0][0] = dot_product(($inputs)[2], $d_relu[2]) +
        dot_product(($inputs)[1], $d_relu[1]);

      $dweights_l1[0][1] = dot_product(($inputs)[2], transpose($d_relu)[1]) +
        dot_product(($inputs)[1], transpose($d_relu)[0]);

      $dweights_l1[1][0] + dot_product(($inputs)[3], $d_relu[1]) +
        dot_product(($inputs)[0], $d_relu[3]);

      $dweights_l1[1][1] + dot_product(($inputs)[3], transpose($d_relu)[1]) +
        dot_product(($inputs)[0], transpose($d_relu)[0]);

      /* 
       * hidden1 derivatives in respect to each input (derivatives * weights.T)
       */
      for ($z = 0; $z < count($inputs); $z++) {
        $d_row = [];
        for ($i = 0; $i < count($this->hidden1_weights); $i++) {
          $d_row[] = dot_product($d_relu[$z], $this->hidden1_weights[$i]);
        }
        $dinputs_l1[] = $d_row;
      }

      /* 
       * hidden1 derivatives in respect to bias
       */
      for ($z = 0; $z < count($d_relu); $z++) {
        $dbias_l1[0] += $d_relu[$z][0];
        $dbias_l1[1] += $d_relu[$z][1];
      }

      /* 
       * optimizing hidden2 layer
       */
      for ($i = 0; $i < count($this->hidden2_weights); $i++) {
        $this->hidden2_weights[$i][0] += -$this->learningRate * $dweights_l2[$i][0];
        $this->hidden2_weights[$i][1] += -$this->learningRate * $dweights_l2[$i][1];
      }

      for ($i = 0; $i < count($this->hidden2_bias); $i++) {
        $this->hidden2_bias[$i] += -$this->learningRate * $dbias_l2[$i];
      }

      /*  
       * optimizing hidden1 layer
       */
      for ($i = 0; $i < count($this->hidden1_weights); $i++) {
        $this->hidden1_weights[$i][0] += -$this->learningRate * $dweights_l1[$i][0];
        $this->hidden1_weights[$i][1] += -$this->learningRate * $dweights_l1[$i][1];
      }

      for ($i = 0; $i < count($this->hidden1_bias); $i++) {
        $this->hidden1_bias[$i] += -$this->learningRate * $dbias_l1[$i];
      }

      echo ("acc: " . $accuracy . ", loss: " . $loss . "\n");

      $e++;
    }
  }
}

function dot_product($v1, $v2)
{
  return array_sum(
    array_map(
      function ($a, $b) {
        return $a * $b;
      },
      $v1,
      $v2
    )
  );
}

function ReLU($x)
{
  return max(0, $x);
}

function softmax(array $x)
{
  $sum = 0;

  foreach ($x as &$val) {
    $val = exp($val);
    $sum += $val;
  }

  return array_map(function ($val) use ($sum) {
    return $val / $sum;
  }, $x);
}

function categ_cross_entropy($x, $y)
{
  $confidences = [];
  for ($i = 0; $i < count($x); $i++) {
    $confidences[] = dot_product($x[$i], $y[$i]);
  }

  $negative_logs = [];
  foreach ($confidences as $x) {
    $negative_logs[] = -log($x);
  }

  return (array_sum($negative_logs) / count($negative_logs));
}

function categ_cross_entropy_d($x, $y)
{
  $samples = count($x);

  $dinputs = $x;

  for ($z = 0; $z < count($x); $z++) {
    $y_true = array_keys($y[$z], max($y[$z]))[0];
    $dinputs[$z][$y_true] -= 1;
  }

  $dinputs_normalized = [];
  foreach ($dinputs as $y) {
    $d_row = [];
    foreach ($y as $z) {
      $d_row[] = ($z / $samples);
    }
    $dinputs_normalized[] = $d_row;
  }

  return $dinputs_normalized;
}

function transpose($input)
{
  $out = [];
  foreach ($input as $key => $subarrary) {
    foreach ($subarrary as $subkey => $subvalue) {
      $out[$subkey][$key] = $subvalue;
    }
  }
  return $out;
}


$nn = new NN();

# acc -> 0.25
// var_dump($nn->predict([1, 1])); # miss
// var_dump($nn->predict([0, 0])); # miss
// var_dump($nn->predict([1, 0])); # miss
// var_dump($nn->predict([0, 1])); # hit

// echo ("\n");
// echo ("\n");

# Fitting XOR function
# 1 / 1 = 0
# 0 / 0 = 0
# 1 / 0 = 1
# 0 / 1 = 1
$nn->fit([[1, 1], [0, 0], [1, 0], [0, 1]], [[0, 1], [0, 1], [1, 0], [1, 0]]);

// echo ("\n");
// echo ("\n");

# acc -> 1.0
// var_dump($nn->predict([1, 1])); # hit
// var_dump($nn->predict([0, 0])); # hit 
// var_dump($nn->predict([1, 0])); # hit
// var_dump($nn->predict([0, 1])); # hit
