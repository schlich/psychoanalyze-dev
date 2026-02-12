Feature: Streaming Bayesian Updates
  As a researcher analyzing psychophysical data
  I want to update posterior estimates incrementally as trials arrive
  So that I can monitor threshold estimation in real time

  Background:
    Given a streaming configuration with a chosen inference method
    And an initialised streaming state

  Scenario: Initialise SMC streaming estimator
    Given the inference method is "SMC"
    And the particle count is 1000
    When the streaming estimator is initialised
    Then the state should contain 1000 particles
    And the effective sample size should equal the particle count

  Scenario: Initialise variational streaming estimator
    Given the inference method is "ADVI"
    When the streaming estimator is initialised
    Then the state should contain variational parameters
    And the step count should be 0

  Scenario: Incorporate a single trial via SMC
    Given the inference method is "SMC"
    And the streaming estimator is initialised
    When a trial with intensity 0.5 and response 1 is observed
    Then the step count should increment by 1
    And the effective sample size should be tracked

  Scenario: Posterior concentrates after repeated observations
    Given the inference method is "SMC"
    And the particle count is 500
    And the streaming estimator is initialised
    When 20 trials with hits above threshold and misses below are observed
    Then the threshold estimate should shift toward the true threshold

  Scenario: Summarise current posterior
    Given the inference method is "SMC"
    And the streaming estimator is initialised
    When the posterior is summarised
    Then the summary should contain threshold, slope, ESS, and step count
    And all summary values should be finite

  Scenario: Switch between SMC and variational backends
    Given the inference method is "ADVI"
    And the streaming estimator is initialised
    When a trial with intensity 0.0 and response 1 is observed
    Then the variational means should remain finite
    And the variational standard deviations should remain finite
