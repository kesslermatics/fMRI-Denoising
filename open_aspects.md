# Open Aspects

## Artificial noising
- Should we simulate different "kinds of noise" based on real world effects like head movement or just add random noise?
- If we simulate noise from what would be real world processes we might also need to simulate additional data like a reference volume for volume registration right (that would usually come directly out of many MRI devices)?
- Would this be a way to simulate temporal noise also or what is considered temporal noise in general and would it make sense to also use random noise for that?
- Is there a requirement on how "bad" the data has to be after adding any kind of noise?

## Data Processing
- Does it make sense to use classic denoising preprocessing techniques like volume registration in our project? Since basically due to the artificially constructed real world like noise which we would then mitigate again by this kind of preprocessing, the question arises why to even do this in the first place and not just add random noise?

## Modeling
- Nothing open so far
