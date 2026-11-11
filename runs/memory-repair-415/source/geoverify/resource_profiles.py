"""Measured resource profiles; unknown shapes keep explicit legacy handling."""
NORMAL_SHAPES={(222,296),(480,640),(512,512)}
RETAINED_SHAPES={(600,960),(768,1024)}
def profile_for_shape(hw):
    hw=tuple(hw)
    if hw in NORMAL_SHAPES:return 'normal9g'
    if hw in RETAINED_SHAPES:return 'retained16g'
    return 'default'
def validate_profile_window(profile,shape):
    if profile not in ('default','normal9g','retained16g'):raise ValueError('Unknown resource profile')
    if profile!='default' and (shape[0]!=8 or profile_for_shape(shape[1:3])!=profile):
        raise ValueError('Resource profile/window shape mismatch')
