import React from 'react';
import styles from '../styles/Chat.module.css';

const EXAMPLES = [
  '비염이 있는데 눈이 따가워요.',
  '알러지 때문에 비염이 심한데 세노바퀵 먹어도 될까요?',
  '두통에 타이레놀 먹어도 될까요?',
];

const EmptyState: React.FC = () => {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>질문을 입력해 주세요</h2>
      <p className={styles.emptyDesc}>
        <code>rag_qna_multi.py</code>의 직접 입력 모드와 같이, 증상만 말하거나 약 이름을
        포함해 질문할 수 있습니다. 역질문이 나오면 예/아니오로 답해 주세요.
      </p>
      <ul className={styles.exampleList}>
        {EXAMPLES.map((example) => (
          <li key={example}>{example}</li>
        ))}
      </ul>
    </div>
  );
};

export default EmptyState;
