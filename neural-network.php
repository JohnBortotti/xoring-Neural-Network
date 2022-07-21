<?php

class NN
{
  public $epochs = 10;

  # starting weights (usually random)
  public $hidden1_weights = [[0.00905953, -0.00076626], [-0.02411565,  0.00856424]];
  public $hidden1_bias = [0.0, 0.0];

  public $hidden2_weights = [[0.01191972,  0.0147446], [0.00220843, -0.0027133]];
  public $hidden2_bias = [0.0, 0.0];

  public $learningRate = 1.0;

  public function predict($input)
  {
    # dot product hidden1 layer
    $dot_hidden_1 = $this->hidden1_bias;

    $dot_hidden_1[0] += $input[0] * $this->hidden1_weights[0][0];
    $dot_hidden_1[0] += $input[1] * $this->hidden1_weights[1][0];

    $dot_hidden_1[1] += $input[0] * $this->hidden1_weights[0][1];
    $dot_hidden_1[1] += $input[1] * $this->hidden1_weights[1][1];

    # ReLU activation
    $dot_hidden_1_relu = [];
    $dot_hidden_1_relu[0] = ReLU($dot_hidden_1[0]);
    $dot_hidden_1_relu[1] = ReLU($dot_hidden_1[1]);

    # dot product hidden2 layer
    $dot_hidden_2 = $this->hidden2_bias;

    $dot_hidden_2[0] += $dot_hidden_1_relu[0] * $this->hidden2_weights[0][0];
    $dot_hidden_2[0] += $dot_hidden_1_relu[1] * $this->hidden2_weights[1][0];

    $dot_hidden_2[1] += $dot_hidden_1_relu[0] * $this->hidden2_weights[0][1];
    $dot_hidden_2[1] += $dot_hidden_1_relu[1] * $this->hidden2_weights[1][1];

    # softmax activation
    return softmax($dot_hidden_2);
  }

  public function fit($inputs, $labels)
  {
    $e = 0;
    while ($e < $this->epochs) {
      # dot product hidden1 layer
      $dot_hidden_1 = [];

      foreach ($inputs as $input) {
        $d_row = $this->hidden1_bias;

        for ($i = 0; $i <= count($d_row) - 1; $i++) {
          $d_row[$i] += $input[0] * $this->hidden1_weights[0][$i];
          $d_row[$i] += $input[1] * $this->hidden1_weights[1][$i];
        }

        $dot_hidden_1[] = $d_row;
      }

      # ReLU activation
      $dot_hidden_1_relu = array_map(function ($x) {
        $y = [];
        foreach ($x as $z) {
          $y[] = ReLU($z);
        }

        return $y;
      }, $dot_hidden_1);

      # dot product hidden1 layer
      $dot_hidden_2 = [];
      foreach ($dot_hidden_1_relu as $input) {
        $d_row = $this->hidden2_bias;

        for ($i = 0; $i <= count($d_row) - 1; $i++) {
          $d_row[$i] += $input[0] * $this->hidden2_weights[0][$i];
          $d_row[$i] += $input[1] * $this->hidden2_weights[1][$i];
        }

        $dot_hidden_2[] = $d_row;
      }

      # softmax activation
      $softmax_output = [];
      foreach ($dot_hidden_2 as $input) {
        $softmax_output[] = softmax($input);
      }

      # calculating loss
      $loss = categ_cross_entropy($softmax_output, $labels);

      # calculating accuracy
      $predictions = [];
      foreach ($softmax_output as $out) {
        $predictions[] = array_keys($out, max($out))[0];
      }

      $y_true = [];
      foreach ($labels as $label) {
        $y_true[] = array_keys($label, max($label))[0];
      }

      $accuracy = [];
      for ($i = 0; $i <= count($predictions) - 1; $i++) {
        $accuracy[] = $predictions[$i] == $y_true[$i];
      }
      $accuracy = array_sum($accuracy) / count($accuracy);

      # backpropagation
      # loss gradient backpropagation

      # loss derivative
      $d_loss = categ_cross_entropy_d($softmax_output, $labels);

      # hidden2 derivatives
      $dweights_l2 = [[0, 0], [0, 0]];
      $dinputs_l2 = [[0, 0], [0, 0], [0, 0], [0, 0]];
      $dbias_l2 = [0, 0];

      # hidden2 derivatives in respect to each weight (inputs * derivatives)
      $dweights_l2[0][0] += $dot_hidden_1_relu[2][0] * $d_loss[2][0];
      $dweights_l2[0][0] += $dot_hidden_1_relu[2][1] * $d_loss[2][1];
      $dweights_l2[0][0] += $dot_hidden_1_relu[1][0] * $d_loss[1][0];
      $dweights_l2[0][0] += $dot_hidden_1_relu[1][1] * $d_loss[1][1];

      $dweights_l2[0][1] += $dot_hidden_1_relu[2][0] * $d_loss[2][1];
      $dweights_l2[0][1] += $dot_hidden_1_relu[2][1] * $d_loss[2][0];
      $dweights_l2[0][1] += $dot_hidden_1_relu[1][0] * $d_loss[1][1];
      $dweights_l2[0][1] += $dot_hidden_1_relu[1][1] * $d_loss[1][0];

      $dweights_l2[1][0] += $dot_hidden_1_relu[3][0] * $d_loss[3][1];
      $dweights_l2[1][0] += $dot_hidden_1_relu[3][1] * $d_loss[3][0];
      $dweights_l2[1][0] += $dot_hidden_1_relu[0][0] * $d_loss[0][1];
      $dweights_l2[1][0] += $dot_hidden_1_relu[0][1] * $d_loss[0][0];

      $dweights_l2[1][1] += $dot_hidden_1_relu[3][0] * $d_loss[3][0];
      $dweights_l2[1][1] += $dot_hidden_1_relu[3][1] * $d_loss[3][1];
      $dweights_l2[1][1] += $dot_hidden_1_relu[0][0] * $d_loss[0][0];
      $dweights_l2[1][1] += $dot_hidden_1_relu[0][1] * $d_loss[0][1];

      # hidden2 derivatives in respect to each input (derivatives * weights)
      $dinputs_l2[0][0] += $this->hidden2_weights[0][0] * $d_loss[0][0];
      $dinputs_l2[0][0] += $this->hidden2_weights[0][1] * $d_loss[0][1];
      $dinputs_l2[0][1] += $this->hidden2_weights[1][0] * $d_loss[0][0];
      $dinputs_l2[0][1] += $this->hidden2_weights[1][1] * $d_loss[0][1];

      $dinputs_l2[1][0] += $this->hidden2_weights[0][0] * $d_loss[1][0];
      $dinputs_l2[1][0] += $this->hidden2_weights[0][1] * $d_loss[1][1];
      $dinputs_l2[1][1] += $this->hidden2_weights[1][0] * $d_loss[1][0];
      $dinputs_l2[1][1] += $this->hidden2_weights[1][1] * $d_loss[1][1];

      $dinputs_l2[2][0] += $this->hidden2_weights[0][0] * $d_loss[2][0];
      $dinputs_l2[2][0] += $this->hidden2_weights[0][1] * $d_loss[2][1];
      $dinputs_l2[2][1] += $this->hidden2_weights[1][0] * $d_loss[2][0];
      $dinputs_l2[2][1] += $this->hidden2_weights[1][1] * $d_loss[2][1];

      $dinputs_l2[3][0] += $this->hidden2_weights[0][0] * $d_loss[3][0];
      $dinputs_l2[3][0] += $this->hidden2_weights[0][1] * $d_loss[3][1];
      $dinputs_l2[3][1] += $this->hidden2_weights[1][0] * $d_loss[3][0];
      $dinputs_l2[3][1] += $this->hidden2_weights[1][1] * $d_loss[3][1];

      # hidden2 derivatives in respect to bias
      for ($z = 0; $z <= count($d_loss) - 1; $z++) {
        $dbias_l2[0] += $d_loss[$z][0];
        $dbias_l2[1] += $d_loss[$z][1];
      }

      # ReLU derivative
      $d_relu = [];
      for ($i = 0; $i <= count($dot_hidden_1) - 1; $i++) {
        $w0 = $dot_hidden_1[$i][0] <= 0 ? 0 : $dinputs_l2[$i][0];
        $w1 = $dot_hidden_1[$i][1] <= 0 ? 0 : $dinputs_l2[$i][1];

        $d_relu[] = [$w0, $w1];
      }

      # hidden2 derivatives
      $dweights_l1 = [[0, 0], [0, 0]];
      $dinputs_l1 = [];
      $dbias_l1 = [0, 0];

      # hidden1 derivatives in respect to each weight (inputs * derivatives)
      $dweights_l1[0][0] += $inputs[2][0] * $d_relu[2][0];
      $dweights_l1[0][0] += $inputs[2][1] * $d_relu[2][1];
      $dweights_l1[0][0] += $inputs[1][0] * $d_relu[1][0];
      $dweights_l1[0][0] += $inputs[1][1] * $d_relu[1][1];

      $dweights_l1[0][1] += $inputs[2][0] * $d_relu[0][1];
      $dweights_l1[0][1] += $inputs[2][1] * $d_relu[0][0];
      $dweights_l1[0][1] += $inputs[1][0] * $d_relu[2][0];
      $dweights_l1[0][1] += $inputs[1][1] * $d_relu[2][1];

      $dweights_l1[1][0] += $inputs[3][0] * $d_relu[1][1];
      $dweights_l1[1][0] += $inputs[3][1] * $d_relu[1][0];
      $dweights_l1[1][0] += $inputs[1][0] * $d_relu[3][1];
      $dweights_l1[1][0] += $inputs[1][1] * $d_relu[3][0];

      $dweights_l1[1][1] += $inputs[3][0] * $d_relu[0][0];
      $dweights_l1[1][1] += $inputs[3][1] * $d_relu[0][1];
      $dweights_l1[1][1] += $inputs[0][0] * $d_relu[3][0];
      $dweights_l1[1][1] += $inputs[0][1] * $d_relu[3][1];

      # hidden1 derivatives in respect to each input (derivatives * weights)
      for ($z = 0; $z <= count($inputs) - 1; $z++) {
        $d_row = [];
        for ($i = 0; $i <= count($this->hidden1_weights) - 1; $i++) {
          $d_row[] = dot_product($d_relu[$z], $this->hidden1_weights[$i]);
        }
        $dinputs_l1[] = $d_row;
      }

      # hidden1 derivatives in respect to bias
      for ($z = 0; $z <= count($d_relu) - 1; $z++) {
        $dbias_l1[0] += $d_relu[$z][0];
        $dbias_l1[1] += $d_relu[$z][1];
      }

      # optimizing hidden2
      $this->hidden2_weights[0][0] += -$this->learningRate * $dweights_l2[0][0];
      $this->hidden2_weights[0][1] += -$this->learningRate * $dweights_l2[0][1];
      $this->hidden2_weights[1][0] += -$this->learningRate * $dweights_l2[1][0];
      $this->hidden2_weights[1][1] += -$this->learningRate * $dweights_l2[1][1];
      $this->hidden2_bias[0] += -$this->learningRate * $dbias_l2[0];
      $this->hidden2_bias[1] += -$this->learningRate * $dbias_l2[1];

      ## optimizing hidden1
      $this->hidden1_weights[0][0] += -$this->learningRate * $dweights_l1[0][0];
      $this->hidden1_weights[0][1] += -$this->learningRate * $dweights_l1[0][1];
      $this->hidden1_weights[1][0] += -$this->learningRate * $dweights_l1[1][0];
      $this->hidden1_weights[1][1] += -$this->learningRate * $dweights_l1[1][1];
      $this->hidden1_bias[0] += -$this->learningRate * $dbias_l1[0];
      $this->hidden1_bias[1] += -$this->learningRate * $dbias_l1[1];

      echo ("acc: " . $accuracy . ", loss: " . $loss . "\n");
      echo ("\n");

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
  for ($i = 0; $i <= count($x) - 1; $i++) {
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

  for ($z = 0; $z <= count($x) - 1; $z++) {
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
