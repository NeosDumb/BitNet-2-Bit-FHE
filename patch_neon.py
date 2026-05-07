import re

with open('src/ggml-bitnet-mad.cpp', 'r') as f:
    code = f.read()

# 1. Replace the accu32/accula variable declarations to remove them or comment them out
code = re.sub(
    r'int16x8_t (accu32|accula)(\[\w+\])?\s*=\s*vdupq_n_s16\(0\);',
    r'// \g<0>',
    code
)

code = re.sub(
    r'int16x8_t (accu32|accula)\[PARALLEL_SIZE\];',
    r'// \g<0>',
    code
)

code = re.sub(
    r'for\s*\(int \w+\s*=\s*0;\s*\w+\s*<\s*PARALLEL_SIZE;\s*\w+\+\+\)\s*\{\s*(accu32|accula)\[\w+\]\s*=\s*vdupq_n_s16\(0\);\s*\}',
    r'/* \g<0> */',
    code
)


# 2. Replace the vmlal_s8 sections
# In the loops, we have:
# accu32 = vmlal_s8(accu32, vget_low_s8(q8_0), vget_low_s8(yq8_0));
# ... up to q8_3
# Sometimes it's accu32[rb], accu32[iy], accula, accula[rb], accula[iy]

def replacer_vmlal(match):
    acc_var = match.group(1) # e.g. accu32, accu32[rb], accula
    # Find the corresponding accu variable (usually accu, accu[rb], accu[iy])
    # The corresponding one can be deduced from the acc_var: if it has [xx], accu has [xx] too
    m = re.search(r'\[(.*?)\]', acc_var)
    if m:
        target_acc = f"accu[{m.group(1)}]"
    else:
        target_acc = "accu"

    return f"""// Mathematical Optimization: SIMD Pairwise Reduction
                    // Since q8 in I2_S is strictly in {{-1, 0, 1}} and yq8 is clamped to [-127, 127],
                    // their product vmulq_s8 will strictly bound within [-127, 127], avoiding 8-bit overflow.
                    // Thus, we can completely bypass vmlal_s8 (which requires high/low unpacking and extra registers).
                    // Instead, we multiply natively in 8-bit and use pairwise accumulation via vpaddlq_s8 and vpadalq_s16.
                    {target_acc} = vpadalq_s16({target_acc}, vpaddlq_s8(vmulq_s8(q8_0, yq8_0)));
                    {target_acc} = vpadalq_s16({target_acc}, vpaddlq_s8(vmulq_s8(q8_1, yq8_1)));
                    {target_acc} = vpadalq_s16({target_acc}, vpaddlq_s8(vmulq_s8(q8_2, yq8_2)));
                    {target_acc} = vpadalq_s16({target_acc}, vpaddlq_s8(vmulq_s8(q8_3, yq8_3)));"""

code = re.sub(
    r'(accu32(?:\[[^\]]+\])?|accula(?:\[[^\]]+\])?)\s*=\s*vmlal_s8\(\1,\s*vget_low_s8\(q8_0\),\s*vget_low_s8\(yq8_0\)\);.*?\1\s*=\s*vmlal_s8\(\1,\s*vget_high_s8\(q8_3\),\s*vget_high_s8\(yq8_3\)\);',
    replacer_vmlal,
    code,
    flags=re.DOTALL
)

# 3. Remove the post-loop merge where accu32/accula is added to accu
# Example:
#             accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32)));
#             accu = vaddq_s32(accu, vmovl_high_s16(accu32));
code = re.sub(
    r'(accu(?:\[[^\]]+\])?)\s*=\s*vaddq_s32\(\1,\s*vmovl_s16\(vget_low_s16\((accu32|accula)(?:\[[^\]]+\])?\)\)\);\s*\1\s*=\s*vaddq_s32\(\1,\s*vmovl_high_s16\(\2(?:\[[^\]]+\])?\)\);',
    r'/* \g<0> */',
    code,
    flags=re.DOTALL
)

# And another format:
# accu[iy] = vaddq_s32(accu[iy], vaddq_s32(vmovl_high_s16(accula[iy]), vmovl_s16(vget_low_s16(accula[iy]))));
code = re.sub(
    r'(accu(?:\[[^\]]+\])?)\s*=\s*vaddq_s32\(\1,\s*vaddq_s32\(vmovl_high_s16\((accu32|accula)(?:\[[^\]]+\])?\),\s*vmovl_s16\(vget_low_s16\(\2(?:\[[^\]]+\])?\)\)\)\);',
    r'/* \g<0> */',
    code,
    flags=re.DOTALL
)


with open('src/ggml-bitnet-mad.cpp', 'w') as f:
    f.write(code)
